import { motion } from "framer-motion"
import { useStore } from "./store"

import AudioUpload from "./components/AudioUploader"
import AudioPlayer from "./components/AudioPlayer"
import Waveform from "./components/Waveform"
import Spectrogram from "./components/Spectrogram"
import PredictionResults from "./components/Predictions"
import FeatureMapExplorer from "./components/FeatureMapExplorer"
import ConfidenceChart from "./components/ConfidenceChart"
import SectionCard from "./components/SectionCard"

export default function App() {
  const { audioURL, result, loading } = useStore()

  const confidenceData = result?.predictions?.map((p: any) => ({class: p.class,
    confidence: Number(p["confidence/accuracy"] ?? 0),
  })) ?? []


  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-black text-white">

      <div className="max-w-[1600px] mx-auto p-8 space-y-8">

        {/* ================= HEADER ================= */}
        <motion.div
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-slate-800/60 backdrop-blur rounded-2xl p-8 shadow-2xl border border-slate-700"
        >
          <h1 className="text-4xl font-bold tracking-tight">
            AI Audio Classification Dashboard
          </h1>

          <p className="text-slate-400 mt-2">
            Real-time CNN inference • Feature visualization • Research-grade analysis
          </p>
        </motion.div>

        {/* ================= UPLOAD ================= */}
        <SectionCard title="Upload Audio">
          <AudioUpload />
        </SectionCard>

        {/* ================= PLAYER ================= */}
        {audioURL && (
          <SectionCard title="Audio Player">
            <AudioPlayer url={audioURL} />
          </SectionCard>
        )}

        {/* ================= LOADING ================= */}
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-sky-400 text-lg"
          >
            🔥 Running model inference…
          </motion.div>
        )}

        {/* ================= RESULTS ================= */}
        {result && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="grid grid-cols-3 gap-8"
          >
            {/* ========= LEFT SIDE ========= */}
            <div className="col-span-2 space-y-8">

              <SectionCard title="Waveform">
                <Waveform audioUrl={audioURL!} />
              </SectionCard>

              <SectionCard title="Spectrogram">
                <Spectrogram data={result.input_spectogram.value} />
              </SectionCard>


              <SectionCard title="Model Confidence">
                <ConfidenceChart data={confidenceData} 
                />
              </SectionCard>


            </div>

            {/* ========= RIGHT SIDE ========= */}
            <div className="space-y-8">

              <SectionCard title="Prediction Results">
                <PredictionResults data={result.predictions} />
              </SectionCard>

              <SectionCard title="Feature Map Explorer">
                <FeatureMapExplorer data={result.visualization} />
              </SectionCard>

            </div>
          </motion.div>
        )}

      </div>
    </div>
  )
}
