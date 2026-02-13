//import FeatureMap from "./FeatureMaps"

interface Props {
  data: Record<string, { value: number[][] }>
}

export default function FeatureMapExplorer({ data }: Props) {
  return (
    <div className="grid grid-cols-4 gap-2">
      {Object.entries(data).map(([name, map]) => (
        <div key={name} className="text-center">
          <FeatureMap data={map.value} />
          <p className="text-xs text-slate-400">{name}</p>
        </div>
      ))}
    </div>
  )
}

function FeatureMap({ data }: { data: number[][] }) {
  const h = data.length
  const w = data[0].length
  const max = Math.max(...data.flat().map(Math.abs)) || 1

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      className="w-full h-24 rounded"
    >
      {data.map((row, i) =>
        row.map((v, j) => {
          const n = Math.abs(v) / max
          const color = `rgb(${0}, ${150 * n}, ${255})`

          return (
            <rect
              key={`${i}-${j}`}
              x={j}
              y={i}
              width={1}
              height={1}
              fill={color}
            />
          )
        })
      )}
    </svg>
  )
}

