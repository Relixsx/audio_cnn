import type { ReactNode } from "react"
import { motion } from "framer-motion"

interface Props {
  title: string
  children: ReactNode
  right?: ReactNode
}

export default function SectionCard({ title, children, right }: Props) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className="
        relative rounded-2xl
        bg-slate-800/70 backdrop-blur-xl
        border border-slate-700
        p-6 shadow-xl
      "
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white">{title}</h2>
        {right}
      </div>

      {/* Content */}
      <div className="text-slate-200">{children}</div>
    </motion.section>
  )
}
