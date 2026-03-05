import { useEffect, useRef } from "react";

interface AudioVisualizerProps {
  analyserNode: AnalyserNode | null;
  barCount?: number;
  className?: string;
}

export function AudioVisualizer({
  analyserNode,
  barCount = 20,
  className,
}: AudioVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (!analyserNode) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const dataArray = new Uint8Array(analyserNode.frequencyBinCount);

    const draw = () => {
      rafRef.current = requestAnimationFrame(draw);

      analyserNode.getByteFrequencyData(dataArray);

      const { width, height } = canvas;
      ctx.clearRect(0, 0, width, height);

      const step = Math.floor(dataArray.length / barCount);
      const barWidth = width / barCount;
      const gap = 2;

      for (let i = 0; i < barCount; i++) {
        const value = dataArray[i * step] / 255;
        const barHeight = value * height;
        const x = i * barWidth + gap / 2;
        const y = height - barHeight;

        ctx.fillStyle = `hsl(${210 + value * 60}, 80%, ${40 + value * 20}%)`;
        ctx.fillRect(x, y, barWidth - gap, barHeight);
      }
    };

    draw();

    return () => {
      cancelAnimationFrame(rafRef.current);
    };
  }, [analyserNode, barCount]);

  return (
    <canvas
      ref={canvasRef}
      width={200}
      height={60}
      className={className}
    />
  );
}
