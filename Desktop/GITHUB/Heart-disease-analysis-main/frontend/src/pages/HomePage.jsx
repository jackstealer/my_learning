import { motion } from 'framer-motion'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { Link } from 'react-router-dom'
import { 
  Brain, 
  Activity, 
  Shield, 
  Zap, 
  FileText, 
  Users,
  CheckCircle,
  TrendingUp,
  Award,
  Lock
} from 'lucide-react'
import Heart3D from '../components/Heart3D'
import AnimatedCounter from '../components/AnimatedCounter'
import GlowingCard from '../components/GlowingCard'
import PulsingHeart from '../components/PulsingHeart'
import ParticleBackground from '../components/ParticleBackground'

const HomePage = () => {
  const features = [
    {
      icon: <Brain className="w-8 h-8" />,
      title: 'Data-Driven AI',
      description: '100% data-driven model with zero hardcoded thresholds. All risk factors extracted from real clinical data.'
    },
    {
      icon: <Activity className="w-8 h-8" />,
      title: 'Risk Stratification',
      description: 'Advanced risk scoring with probability-based predictions and detailed clinical reasoning.'
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: 'HIPAA Compliant',
      description: 'Enterprise-grade security with encrypted data transmission and privacy protection.'
    },
    {
      icon: <Zap className="w-8 h-8" />,
      title: 'Real-Time Analysis',
      description: 'Instant risk assessment with sub-second response times via RESTful API.'
    },
    {
      icon: <FileText className="w-8 h-8" />,
      title: 'Detailed Reports',
      description: 'Comprehensive risk reports with identified factors and clinical documentation.'
    },
    {
      icon: <Users className="w-8 h-8" />,
      title: 'Clinical Validation',
      description: 'Validated against established cardiovascular risk models by healthcare professionals.'
    }
  ]

  const stats = [
    { value: '4,500+', label: 'Training Records' },
    { value: '17', label: 'Clinical Features' },
    { value: '95%+', label: 'Accuracy' },
    { value: '100%', label: 'Data-Driven' }
  ]

  const trustBadges = [
    { icon: <Award />, title: 'Clinically Validated', desc: 'Validated against established models' },
    { icon: <Lock />, title: 'HIPAA Compliant', desc: 'Healthcare data privacy standards' },
    { icon: <CheckCircle />, title: 'Open Source', desc: 'Transparent algorithms' },
    { icon: <Users />, title: 'Professional Support', desc: 'Technical support available' }
  ]

  return (
    <div className="pt-20">
      {/* Particle Background */}
      <ParticleBackground />
      
      {/* Hero Section with 3D Heart */}
      <section className="relative min-h-screen flex items-center overflow-hidden bg-gradient-to-br from-blue-50 via-white to-cyan-50">
        {/* 3D Background */}
        <div className="absolute inset-0 opacity-30">
          <Canvas>
            <PerspectiveCamera makeDefault position={[0, 0, 5]} />
            <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={0.5} />
            <Heart3D />
          </Canvas>
        </div>

        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
            >
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="inline-flex items-center space-x-2 bg-green-100 text-green-700 px-4 py-2 rounded-full mb-6"
              >
                <PulsingHeart size="sm" color="green" />
                <span className="font-semibold">AI Model Active</span>
              </motion.div>

              <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold text-slate-900 mb-6 leading-tight">
                Professional
                <span className="block text-transparent bg-clip-text bg-gradient-to-r from-primary-600 to-secondary-500">
                  Cardiovascular
                </span>
                Risk Assessment
              </h1>

              <p className="text-xl text-gray-600 mb-8 leading-relaxed">
                Advanced AI-powered risk stratification using data-driven algorithms trained on 
                real-world clinical data. Designed for healthcare professionals and research institutions.
              </p>

              <div className="flex flex-wrap gap-4">
                <Link to="/assessment" className="btn-primary text-lg">
                  <Activity className="w-5 h-5 inline mr-2" />
                  Start Assessment
                </Link>
                <Link to="/about" className="btn-secondary text-lg">
                  Learn More
                </Link>
              </div>

              {/* Quick Stats */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-12">
                {stats.map((stat, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 + index * 0.1 }}
                    className="text-center"
                  >
                    <div className="text-3xl font-bold text-primary-600">
                      <AnimatedCounter 
                        end={parseInt(stat.value.replace(/[^0-9]/g, '')) || 0} 
                        suffix={stat.value.includes('+') ? '+' : stat.value.includes('%') ? '%' : ''}
                        duration={2}
                      />
                    </div>
                    <div className="text-sm text-gray-600">{stat.label}</div>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.3 }}
              className="hidden lg:block"
            >
              <div className="relative w-full h-[600px]">
                <Canvas>
                  <PerspectiveCamera makeDefault position={[0, 0, 5]} />
                  <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={1} />
                  <Heart3D />
                </Canvas>
              </div>
            </motion.div>
          </div>
        </div>

        {/* Scroll Indicator */}
        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="absolute bottom-10 left-1/2 transform -translate-x-1/2"
        >
          <div className="w-6 h-10 border-2 border-gray-400 rounded-full flex justify-center">
            <div className="w-1 h-3 bg-gray-400 rounded-full mt-2"></div>
          </div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="section-title">Enterprise-Grade Features</h2>
            <p className="section-subtitle">
              Built for healthcare professionals with clinical accuracy and compliance in mind
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <GlowingCard
                key={index}
                className="card group hover:border-primary-500 transition-all duration-300"
                glowColor="primary"
              >
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                >
                  <div className="w-16 h-16 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-xl flex items-center justify-center text-white mb-4 group-hover:scale-110 transition-transform">
                    {feature.icon}
                  </div>
                  <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
                  <p className="text-gray-600 leading-relaxed">{feature.description}</p>
                </motion.div>
              </GlowingCard>
            ))}
          </div>
        </div>
      </section>

      {/* Trust Indicators */}
      <section className="py-20 bg-slate-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {trustBadges.map((badge, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ scale: 1.05, y: -5 }}
                className="text-center cursor-pointer"
              >
                <motion.div 
                  className="w-20 h-20 bg-primary-100 rounded-full flex items-center justify-center text-primary-600 mx-auto mb-4"
                  whileHover={{ rotate: 360 }}
                  transition={{ duration: 0.6 }}
                >
                  {badge.icon}
                </motion.div>
                <h4 className="font-semibold text-lg mb-2">{badge.title}</h4>
                <p className="text-gray-600 text-sm">{badge.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-primary-600 to-secondary-500 text-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Ready to Assess Cardiovascular Risk?
            </h2>
            <p className="text-xl mb-8 opacity-90">
              Start using our AI-powered risk assessment platform today
            </p>
            <Link to="/assessment" className="inline-block bg-white text-primary-600 px-8 py-4 rounded-lg font-semibold text-lg hover:bg-gray-100 transition-colors shadow-xl">
              <TrendingUp className="w-6 h-6 inline mr-2" />
              Begin Assessment
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  )
}

export default HomePage
