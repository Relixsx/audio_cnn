interface Props {
  data: number[][]
}

export default function Spectrogram({ data }: Props) {
  if (!data || !data.length) {
    return <div className="text-slate-400">No spectrogram data</div>
  }

  const height = data.length
  const width = data[0].length

  const flat = data.flat()
  const max = Math.max(...flat)
  const min = Math.min(...flat)
  const range = max - min || 1

  const normalize = (v: number) => (v - min) / range

  return (
    <div className="w-full overflow-auto bg-black rounded-lg p-2">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="w-full h-[320px]"
      >
        {data.map((row, i) =>
          row.map((v, j) => {
            const n = normalize(v)

            // 🔥 Professional spectrogram colors
            const r = Math.floor(255 * n)
            const g = Math.floor(180 * Math.pow(n, 1.5))
            const b = Math.floor(255 * (1 - n))

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
    </div>
  )
}
