interface Props {
  data: { class: string; "confidence/accuracy": number }[]
}

export default function PredictionResults({ data }: Props) {
  return (
    <div className="space-y-3">
      {data.map((p, i) => (
        <div key={i}>
          <div className="flex justify-between text-sm">
            <span>{p.class}</span>
            <span>{(p["confidence/accuracy"] * 100).toFixed(1)}%</span>
          </div>

          <div className="bg-slate-700 h-2 rounded">
            <div
              className="bg-emerald-400 h-2 rounded"
              style={{
                width: `${p["confidence/accuracy"] * 100}%`,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}
