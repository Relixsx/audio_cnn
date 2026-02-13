import { useEffect, useRef } from "react"
import WaveSurfer from "wavesurfer.js"

interface Props {
  audioUrl: string
}

export default function Waveform({ audioUrl }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return

    const wave = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "#94a3b8",
      progressColor: "#38bdf8",
      height: 120,
    })

    wave.load(audioUrl)

    return () => wave.destroy()
  }, [audioUrl])

  return <div ref={containerRef} />
}
