import { useStore } from "../store"

export default function AudioUpload() {
  const { setAudioURL, setResult, setLoading } = useStore()

  const handleFile = async (file: File) => {
    setAudioURL(URL.createObjectURL(file))
    setLoading(true)

    const buffer = await file.arrayBuffer()
    const base64 = btoa(
      new Uint8Array(buffer)
        .reduce((data, byte) => data + String.fromCharCode(byte), "")
    )

    const res = await fetch(
      "https://relixsx--sound-classification-fastapi-app.modal.run/inference",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio_data: base64 }),
      }
    )

    const json = await res.json()
    setResult(json)
    setLoading(false)
  }

  return (
    <div className="bg-slate-800 rounded-xl p-6 shadow">
      <input
        type="file"
        accept=".wav"
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) handleFile(f)
        }}
      />
    </div>
  )
}
