import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Download, FileUp, Grid, Wand2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { fetchGlycoforms } from '../api';
import { GlycoformRecord } from '../types';
import { createScoreScale, scoreToColor } from '../utils/bindingScore';
import Button from '../components/ui/Button';
import Card from '../components/ui/Card';
import styles from './ExplorerView.module.css';

const safeKey = (value: string) => value.replace(/[^A-Za-z0-9._-]/g, '_');

export default function ExplorerView() {
  const [data, setData] = useState<GlycoformRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    setIsLoading(true);
    fetchGlycoforms()
      .then((records) => setData(records))
      .finally(() => setIsLoading(false));
  }, []);

  const fcgrs = useMemo(() => {
    return Array.from(new Set(data.map((row) => row.fcgr_name))).sort();
  }, [data]);

  const glycans = useMemo(() => {
    return Array.from(new Set(data.map((row) => row.glycan_name))).sort();
  }, [data]);

  const cellMap = useMemo(() => {
    const map = new Map<string, GlycoformRecord>();
    data.forEach((row) => {
      map.set(`${row.fcgr_name}__${row.glycan_name}`, row);
    });
    return map;
  }, [data]);

  const kdValues = useMemo(() => {
    return data
      .map((row) => row.predicted_kd_nm ?? row.binding_kd_nm)
      .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
  }, [data]);

  const scale = useMemo(() => createScoreScale(kdValues), [kdValues]);

  const handleCellClick = (fcgr: string, glycan: string) => {
    const params = new URLSearchParams({ fcgr, glycan });
    navigate(`/compare?${params.toString()}`);
  };

  const handleExportCsv = () => {
    const lines = ['fcgr_name,glycan_name,binding_score'];
    fcgrs.forEach((fcgr) => {
      glycans.forEach((glycan) => {
        const record = cellMap.get(`${fcgr}__${glycan}`);
        if (!record) {
          lines.push(`${fcgr},${glycan},`);
          return;
        }
        const kd = record.predicted_kd_nm ?? record.binding_kd_nm;
        const score = kd ? scale(kd) : 0;
        lines.push(`${fcgr},${glycan},${score.toFixed(3)}`);
      });
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'fcgr_binding_heatmap.csv';
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const handleExportPymol = () => {
    const lines = [
      '# GlycoML batch rendering script',
      'bg_color white',
      'set ray_opaque_background, off',
    ];
    data.forEach((row) => {
      const key = safeKey(`${row.fcgr_name}_${row.glycan_name}`);
      lines.push(`load outputs/phase3_pymol/${key}.pdb, ${key}`);
      lines.push(`png outputs/phase3_pymol/${key}.png, width=1200, height=1200, dpi=300, ray=1`);
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'glycoml_batch_render.pml';
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const handleExportPng = () => {
    if (!canvasRef.current) {
      return;
    }
    const cellSize = 50;
    const scaleFactor = 3;
    const width = glycans.length * cellSize;
    const height = fcgrs.length * cellSize;
    const canvas = canvasRef.current;
    canvas.width = width * scaleFactor;
    canvas.height = height * scaleFactor;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }
    ctx.scale(scaleFactor, scaleFactor);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    fcgrs.forEach((fcgr, rowIndex) => {
      glycans.forEach((glycan, colIndex) => {
        const record = cellMap.get(`${fcgr}__${glycan}`);
        const x = colIndex * cellSize;
        const y = rowIndex * cellSize;
        if (!record) {
          ctx.fillStyle = '#f3f4f6';
          ctx.fillRect(x, y, cellSize, cellSize);
          ctx.strokeStyle = '#e5e7eb';
          ctx.beginPath();
          ctx.moveTo(x, y + cellSize);
          ctx.lineTo(x + cellSize, y);
          ctx.stroke();
          return;
        }
        const kd = record.predicted_kd_nm ?? record.binding_kd_nm;
        const score = kd ? scale(kd) : 0.5;
        ctx.fillStyle = scoreToColor(score);
        ctx.fillRect(x, y, cellSize, cellSize);
        ctx.fillStyle = score > 0.6 ? '#ffffff' : '#0a0a0a';
        ctx.font = '12px Urbanist, sans-serif';
        ctx.fillText(score.toFixed(2), x + 8, y + 28);
      });
    });

    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = 'fcgr_binding_heatmap.png';
    link.click();
  };

  return (
    <div className="app-shell">
      <main className={styles.layout} role="main" id="main-content">
        <section className={styles.header}>
          <div>
            <h2>FcγR Binding Explorer</h2>
            <p className={styles.subhead}>
              Heatmap view of predicted binding scores across FcγR allotypes and glycan variants.
            </p>
          </div>
          <div className={styles.actions}>
            <Button
              variant="secondary"
              onClick={handleExportPng}
              icon={<Download size={16} aria-hidden="true" />}
              disabled={isLoading || data.length === 0}
            >
              Download Heatmap PNG
            </Button>
            <Button
              variant="secondary"
              onClick={handleExportCsv}
              icon={<FileUp size={16} aria-hidden="true" />}
              disabled={isLoading || data.length === 0}
            >
              Export CSV
            </Button>
            <Button
              variant="secondary"
              onClick={handleExportPymol}
              icon={<Wand2 size={16} aria-hidden="true" />}
              disabled={isLoading || data.length === 0}
            >
              Generate PyMOL Script
            </Button>
            <span className={styles.exportHint}>PNG exports render at 300 DPI.</span>
          </div>
        </section>

        <Card className={styles.heatmapCard}>
          <div className={styles.legend}>
            <div className={styles.legendItem}>
              <span className={styles.legendSwatch} style={{ background: '#5771FE' }} />
              Low binding (0.0-0.4)
            </div>
            <div className={styles.legendItem}>
              <span className={styles.legendSwatch} style={{ background: '#F59E0B' }} />
              Medium (0.4-0.7)
            </div>
            <div className={styles.legendItem}>
              <span className={styles.legendSwatch} style={{ background: '#EF4444' }} />
              High (0.7-1.0)
            </div>
            <div className={styles.legendItem}>
              <span className={`${styles.legendSwatch} ${styles.legendMissing}`} />
              Missing data
            </div>
          </div>

          {isLoading ? (
            <div className={styles.loadingGrid} aria-busy="true">
              {Array.from({ length: 36 }).map((_, index) => (
                <div key={`heat-skeleton-${index}`} className="skeleton" />
              ))}
            </div>
          ) : (
            <div className={styles.tableWrapper}>
              <table className={styles.table} aria-label="FcγR binding heatmap">
                <thead>
                  <tr>
                    <th scope="col">FcγR</th>
                    {glycans.map((glycan) => (
                      <th key={glycan} scope="col">
                        {glycan}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {fcgrs.map((fcgr) => (
                    <tr key={fcgr}>
                      <th scope="row">{fcgr}</th>
                      {glycans.map((glycan) => {
                        const record = cellMap.get(`${fcgr}__${glycan}`);
                        const kd = record?.predicted_kd_nm ?? record?.binding_kd_nm;
                        const score = record && kd ? scale(kd) : null;
                        const cellStyle =
                          score !== null
                            ? {
                                background: scoreToColor(score),
                                color: score > 0.6 ? '#ffffff' : '#0a0a0a',
                              }
                            : undefined;
                        return (
                          <td key={`${fcgr}-${glycan}`}>
                            <button
                              type="button"
                              className={`${styles.cell} ${record ? '' : styles.cellMissing}`}
                              style={cellStyle}
                              title={
                                score !== null
                                  ? `${fcgr} · ${glycan}: ${score.toFixed(2)}`
                                  : `${fcgr} · ${glycan}: missing data`
                              }
                              onClick={() => handleCellClick(fcgr, glycan)}
                            >
                              {score !== null ? score.toFixed(2) : '--'}
                            </button>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card className={styles.insightsCard}>
          <div className={styles.insightsHeader}>
            <Grid size={18} color="#5771FE" aria-hidden="true" />
            <span>Explorer Guidance</span>
          </div>
          <ul className={styles.insightsList}>
            <li>Click a cell to open Compare View with the selected FcγR-glycan pair.</li>
            <li>Low binding scores appear in blue; strong binding rises to red.</li>
            <li>Export heatmaps as PNG, CSV, or a PyMOL batch script for offline rendering.</li>
          </ul>
        </Card>
      </main>
      <canvas ref={canvasRef} className={styles.hiddenCanvas} aria-hidden="true" />
    </div>
  );
}
