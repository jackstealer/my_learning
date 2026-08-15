import { motion } from 'framer-motion'
import { useState } from 'react'

const GlowingCard = ({ children, className = '', glowColor = 'primary' }) => {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })
  const [isHovering, setIsHovering] = useState(false)

  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    setMousePosition({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    })
  }

  const glowColors = {
    primary: 'rgba(37, 99, 235, 0.4)',
    success: 'rgba(5, 150, 105, 0.4)',
    danger: 'rgba(220, 38, 38, 0.4)',
    warning: 'rgba(217, 119, 6, 0.4)'
  }

  return (
    <motion.div
      className={`relative overflow-hidden ${className}`}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.3 }}
    >
      {isHovering && (
        <motion.div
          className="absolute pointer-events-none"
          style={{
            left: mousePosition.x,
            top: mousePosition.y,
            width: '300px',
            height: '300px',
            background: `radial-gradient(circle, ${glowColors[glowColor]} 0%, transparent 70%)`,
            transform: 'translate(-50%, -50%)',
          }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        />
      )}
      <div className="relative z-10">
        {children}
      </div>
    </motion.div>
  )
}

export default GlowingCard
