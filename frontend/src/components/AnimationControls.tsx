import React from 'react';

type Props = {
  isPlaying: boolean;
  onToggle: () => void;
};

export default function AnimationControls({ isPlaying, onToggle }: Props) {
  return (
    <div className="panel animation">
      <div className="panel-header">
        <h2>Binding Animation</h2>
        <p>Toggle highlight pulses to emphasize binding effects.</p>
      </div>
      <button className="primary" onClick={onToggle}>
        {isPlaying ? 'Pause animation' : 'Play animation'}
      </button>
    </div>
  );
}
