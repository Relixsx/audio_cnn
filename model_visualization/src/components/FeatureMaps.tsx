interface Props {
  data: number[][]
  title?: string
  internal?: boolean
  spectrogram?: boolean
}

/* ---------- Professional heatmap color scale ---------- */
function getColor(v: number): [number, number, number] {
  // Clamp to [-1, 1]
  const x = Math.max(-1, Math.min(1, v))

  // Blue → Cyan → Yellow → Red
  if (x < -0.5) return [0, 70, 200]
  if (x < 0) return [0, 150, 255]
  if (x < 0.5) return [255, 220, 0]
  return [255, 80, 0]
}

export default function FeatureMap({
  data,
  title = "",
  internal = false,
  spectrogram = false,
}: Props) {
  if (!data || !data.length || !data[0]?.length) return null

  const height = data.length
  const width = data[0].length

  const absMax = Math.max(
    ...data.flat().map((v) => Math.abs(v ?? 0))
  )

  return (
    <div className="w-full text-center">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className={`mx-auto rounded border border-slate-700 bg-black
          ${
            internal
              ? "w-full max-w-32"
              : spectrogram
              ? "w-full max-h-[320px]"
              : "w-full max-w-[520px] max-h-[320px]"
          }`}
      >
        {data.flatMap((row, i) =>
          row.map((value, j) => {
            const norm = absMax === 0 ? 0 : value / absMax
            const [r, g, b] = getColor(norm)

            return (
              <rect
                key={`${i}-${j}`}
                x={j}
                y={i}
                width={1}
                height={1}
                fill={`rgb(${r},${g},${b})`}
              />
            )
          })
        )}
      </svg>

      {title && (
        <p className="mt-2 text-xs text-slate-400">{title}</p>
      )}
    </div>
  )
}
