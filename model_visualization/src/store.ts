import { create } from "zustand"

interface Prediction {
  class: string
  "confidence/accuracy": number
}

interface Result {
  waveform: {
    values: number[]
    duration: number
    sample_rate: number
  }
  input_spectogram: {
    value: number[][]
  }
  predictions: Prediction[]
  visualization: Record<string, any>
}

interface AppState {
  audioURL: string | null
  result: Result | null
  loading: boolean

  setAudioURL: (url: string | null) => void
  setResult: (data: Result | null) => void
  setLoading: (v: boolean) => void
}

export const useStore = create<AppState>((set) => ({
  audioURL: null,
  result: null,
  loading: false,

  setAudioURL: (url) => set({ audioURL: url }),
  setResult: (data) => set({ result: data }),
  setLoading: (v) => set({ loading: v }),
}))
