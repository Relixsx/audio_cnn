import axios from "axios"

const API_URL = "https://relixsx--sound-classification-fastapi-app.modal.run/inference";


export const sendAudio = async (audioBase64: string) => {
  const response = await axios.post(API_URL, {
    audio_data: audioBase64,
  })
  return response.data
}