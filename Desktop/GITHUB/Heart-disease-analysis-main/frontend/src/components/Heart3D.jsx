import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Sphere, MeshDistortMaterial } from '@react-three/drei'
import * as THREE from 'three'

const Heart3D = () => {
  const heartRef = useRef()
  const particlesRef = useRef()

  // Create particles
  const particlesCount = 100
  const positions = new Float32Array(particlesCount * 3)
  
  for (let i = 0; i < particlesCount * 3; i++) {
    positions[i] = (Math.random() - 0.5) * 10
  }

  useFrame((state) => {
    const time = state.clock.getElapsedTime()
    
    // Animate heart
    if (heartRef.current) {
      heartRef.current.rotation.y = time * 0.3
      heartRef.current.position.y = Math.sin(time * 0.5) * 0.2
    }

    // Animate particles
    if (particlesRef.current) {
      particlesRef.current.rotation.y = time * 0.1
      const positions = particlesRef.current.geometry.attributes.position.array
      
      for (let i = 0; i < positions.length; i += 3) {
        positions[i + 1] = Math.sin(time + positions[i]) * 0.5
      }
      
      particlesRef.current.geometry.attributes.position.needsUpdate = true
    }
  })

  return (
    <group>
      {/* Main Heart Shape */}
      <group ref={heartRef}>
        <Sphere args={[1, 64, 64]} position={[0, 0, 0]}>
          <MeshDistortMaterial
            color="#2563eb"
            attach="material"
            distort={0.3}
            speed={2}
            roughness={0.2}
            metalness={0.8}
          />
        </Sphere>
        
        {/* Heart glow */}
        <Sphere args={[1.2, 32, 32]} position={[0, 0, 0]}>
          <meshBasicMaterial
            color="#60a5fa"
            transparent
            opacity={0.2}
            side={THREE.BackSide}
          />
        </Sphere>
      </group>

      {/* Particles */}
      <points ref={particlesRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={particlesCount}
            array={positions}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.05}
          color="#3b82f6"
          transparent
          opacity={0.6}
          sizeAttenuation
        />
      </points>

      {/* Ambient Light */}
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#60a5fa" />
    </group>
  )
}

export default Heart3D
