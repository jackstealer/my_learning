import { motion } from 'framer-motion'
import { Heart } from 'lucide-react'

const PulsingHeart = ({ size = 'md', color = 'red' }) => {
  const sizes = {
    sm: 'w-8 h-8',
    md: 'w-12 h-12',
    lg: 'w-16 h-16',
    xl: 'w-24 h-24'
  }

  const colors = {
    red: 'text-red-500',
    blue: 'text-blue-500',
    green: 'text-green-500',
    purple: 'text-purple-500'
  }

  return (
    <div className="relative inline-block">
      {/* Outer pulse rings */}
      <motion.div
        className={`absolute inset-0 ${colors[color]} opacity-20 rounded-full`}
        animate={{
          scale: [1, 1.5, 1],
          opacity: [0.2, 0, 0.2]
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
      <motion.div
        className={`absolute inset-0 ${colors[color]} opacity-20 rounded-full`}
        animate={{
          scale: [1, 1.8, 1],
          opacity: [0.2, 0, 0.2]
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 0.5
        }}
      />
      
      {/* Heart icon */}
      <motion.div
        animate={{
          scale: [1, 1.1, 1],
        }}
        transition={{
          duration: 1,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      >
        <Heart className={`${sizes[size]} ${colors[color]}`} fill="currentColor" />
      </motion.div>
    </div>
  )
}

export default PulsingHeart
