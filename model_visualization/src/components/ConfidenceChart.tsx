import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from "recharts"

interface Item {
  class: string
  confidence: number
}

interface Props {
  data: Item[]
}

const COLORS = [
  "#22c55e",
  "#38bdf8",
  "#facc15",
  "#fb7185",
  "#a78bfa",
]

export default function ConfidenceChart({ data }: Props) {
  if (!data || data.length === 0) {
    return (
      <div className="text-slate-400 text-sm">
        No prediction data available
      </div>
    )
  }

  const chartData = data.map((d) => ({
    label: d.class,
    confidence: d.confidence,
  }))

  return (
    <div className="w-full h-[320px] min-h-[320px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} layout="vertical">
          
          <XAxis type="number" domain={[0, 1]} hide />

          <YAxis
            type="category"
            dataKey="label"
            width={160}
            tick={{ fill: "#cbd5e1", fontSize: 13 }}
          />

          <Tooltip
            formatter={(v: any) =>
              `${(Number(v) * 100).toFixed(2)}%`
            }
            cursor={{ fill: "rgba(255,255,255,0.05)" }}
          />

          <Bar dataKey="confidence" radius={[0, 8, 8, 0]}>
            {chartData.map((_, i) => (
              <Cell
                key={i}
                fill={COLORS[i % COLORS.length]}
              />
            ))}
          </Bar>

        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
