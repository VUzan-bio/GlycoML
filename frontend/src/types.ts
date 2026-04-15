export type GlycoformRecord = {
  fcgr_name: string;
  glycan_name: string;
  binding_kd_nm: number;
  log_kd: number;
  predicted_log_kd?: number;
  predicted_kd_nm?: number;
  glycan_structure?: string;
  prediction_error?: string | null;
};

export type PredictionRecord = GlycoformRecord & {
  delta_g_kcal_mol?: number;
  affinity_rank?: number;
  affinity_class?: 'strong' | 'moderate' | 'weak' | 'unknown';
  model_version?: string;
  data_version?: string;
  prediction_timestamp?: string;
  structure?: {
    png_path?: string;
    pdb_path?: string;
    pse_path?: string;
    has_glycan?: boolean | null;
    fcgr_name?: string;
    glycan_name?: string;
  };
};
