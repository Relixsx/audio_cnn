interface Props {
  url: string
}

export default function AudioPlayer({ url }: Props) {
  return (
    <div className="bg-slate-800 rounded-xl p-6 shadow">
      <h2 className="text-lg font-semibold mb-2">Audio Player</h2>
      <audio controls src={url} className="w-full" />
    </div>
  )
}
