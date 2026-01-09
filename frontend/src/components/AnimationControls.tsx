import React from 'react';
import { Pause, Play } from 'lucide-react';
import Card from './ui/Card';
import Button from './ui/Button';

type Props = {
  isPlaying: boolean;
  onToggle: () => void;
  isDisabled?: boolean;
};

export default function AnimationControls({ isPlaying, onToggle, isDisabled }: Props) {
  return (
    <Card>
      <div className="section-header">
        <h2>Binding Animation</h2>
        <p>Toggle highlight pulses to emphasize binding affinity.</p>
      </div>
      <Button
        variant="secondary"
        onClick={onToggle}
        disabled={isDisabled}
        icon={isPlaying ? <Pause size={16} aria-hidden="true" /> : <Play size={16} aria-hidden="true" />}
      >
        {isPlaying ? 'Pause animation' : 'Play animation'}
      </Button>
    </Card>
  );
}
