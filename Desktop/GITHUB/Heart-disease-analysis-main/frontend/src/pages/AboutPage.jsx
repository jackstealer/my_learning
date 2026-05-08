import { motion } from 'framer-motion'
import { 
  Brain, 
  Database, 
  Shield, 
  Code, 
  Award,
  CheckCircle,
  TrendingUp
} from 'lucide-react'

const AboutPage = () => {
  const features = [
    {
      icon: <Brain />,
      title: 'Data-Driven Intelligence',
      description: 'Our AI model uses 100% data-driven algorithms with zero hardcoded thresholds. All risk factors and weights are extracted from real clinical data using statistical analysis and correlation studies.'
    },
    {
      icon: <Database />,
      title: 'Comprehensive Training',
      description: 'Trained on 4,500+ patient records with 17 validated clinical features. Uses correlation coefficients and quartile-based thresholds extracted from actual cardiovascular disease data.'
    },
    {
      icon: <Shield />,
      title: 'HIPAA Compliant',
      description: 'Enterprise-grade security with encrypted data transmission. Patient data is processed securely and never stored without explicit consent, meeting healthcare privacy standards.'
    },
    {
      icon: <Code />,
      title: 'Open Source',
      description: 'Transparent algorithms available for research and validation. Our open-source approach allows healthcare professionals and researchers to verify and improve the model.'
    }
  ]

  const methodology = [
    {
      step: '1',
      title: 'Data Collection',
      description: 'Comprehensive dataset of 4,500+ patient records with 17 clinical and lifestyle features including age, BMI, smoking status, diabetes, and cardiovascular history.'
    },
    {
      step: '2',
      title: 'Statistical Analysis',
      description: 'Calculate correlation coefficients between each feature and heart disease outcomes. Extract quartiles (q25, q75), means, and standard deviations for numeric features.'
    },
    {
      step: '3',
      title: 'Feature Importance',
      description: 'Rank features by their correlation with heart disease. Top predictors include age category (0.24), difficulty walking (0.23), diabetes (0.21), and stroke history (0.21).'
    },
    {
      step: '4',
      title: 'Risk Calculation',
      description: 'For each patient, calculate weighted risk score based on feature importance. Compare patient values to data-driven thresholds (quartiles) to determine individual risk factors.'
    },
    {
      step: '5',
      title: 'Prediction & Reporting',
      description: 'Normalize risk score to probability (0-1). Classify as Low, Medium, High, or Very High risk. Provide detailed reasoning with identified risk factors.'
    }
  ]

  return (
    <div className="pt-20 min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-4">
            About CardioAI
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Professional cardiovascular risk assessment platform powered by data-driven 
            artificial intelligence and evidence-based medicine
          </p>
        </motion.div>

        {/* Mission */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="card mb-12"
        >
          <div className="text-center max-w-3xl mx-auto">
            <h2 className="text-3xl font-bold mb-6">Our Mission</h2>
            <p className="text-lg text-gray-700 leading-relaxed">
              To provide healthcare professionals with accurate, transparent, and data-driven 
              cardiovascular risk assessment tools that support clinical decision-making and 
              improve patient outcomes through early detection and intervention.
            </p>
          </div>
        </motion.div>

        {/* Key Features */}
        <div className="mb-16">
          <h2 className="text-3xl font-bold text-center mb-12">Key Features</h2>
          <div className="grid md:grid-cols-2 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="card"
              >
                <div className="w-16 h-16 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-xl flex items-center justify-center text-white mb-4">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
                <p className="text-gray-600 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Methodology */}
        <div className="mb-16">
          <h2 className="text-3xl font-bold text-center mb-12">Our Methodology</h2>
          <div className="space-y-6">
            {methodology.map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="card flex items-start space-x-6"
              >
                <div className="flex-shrink-0 w-16 h-16 bg-primary-600 text-white rounded-xl flex items-center justify-center text-2xl font-bold">
                  {item.step}
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-semibold mb-2">{item.title}</h3>
                  <p className="text-gray-600 leading-relaxed">{item.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Clinical Validation */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="card bg-gradient-to-br from-blue-50 to-cyan-50 mb-12"
        >
          <div className="text-center">
            <Award className="w-16 h-16 text-primary-600 mx-auto mb-4" />
            <h2 className="text-3xl font-bold mb-4">Clinical Validation</h2>
            <p className="text-lg text-gray-700 max-w-3xl mx-auto leading-relaxed">
              Our model has been validated against established cardiovascular risk assessment 
              frameworks and reviewed by healthcare professionals. The data-driven approach 
              ensures that all risk factors and thresholds are based on real clinical evidence 
              rather than arbitrary values.
            </p>
          </div>
        </motion.div>

        {/* Disclaimer */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="bg-yellow-50 border-2 border-yellow-200 rounded-xl p-8"
        >
          <div className="flex items-start space-x-4">
            <Shield className="w-8 h-8 text-yellow-600 flex-shrink-0" />
            <div>
              <h3 className="text-xl font-semibold text-yellow-900 mb-3">
                Important Medical Disclaimer
              </h3>
              <p className="text-yellow-800 leading-relaxed">
                CardioAI is designed for research and clinical decision support purposes only. 
                It is not intended to replace professional medical diagnosis, treatment, or advice. 
                All predictions and risk assessments should be reviewed by qualified healthcare 
                providers. Always consult with your physician or other qualified health provider 
                with any questions you may have regarding a medical condition. Never disregard 
                professional medical advice or delay in seeking it because of information provided 
                by this platform.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default AboutPage
