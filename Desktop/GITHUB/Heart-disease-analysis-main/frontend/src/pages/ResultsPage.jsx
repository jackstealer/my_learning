import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { 
  AlertTriangle, 
  CheckCircle, 
  TrendingUp, 
  Activity,
  FileText,
  Download,
  RotateCcw,
  Heart,
  Lightbulb,
  Target,
  Calendar,
  BookOpen,
  AlertCircle,
  ChevronDown,
  ChevronUp
} from 'lucide-react'
import ProgressRing from '../components/ProgressRing'
import PulsingHeart from '../components/PulsingHeart'
import AnimatedCounter from '../components/AnimatedCounter'

const ResultsPage = () => {
  const [result, setResult] = useState(null)
  const [expandedSections, setExpandedSections] = useState({
    summary: true,
    lifestyle: false,
    medical: false,
    preventive: false,
    education: false
  })
  const navigate = useNavigate()

  useEffect(() => {
    const storedResult = localStorage.getItem('assessmentResult')
    if (storedResult) {
      setResult(JSON.parse(storedResult))
    } else {
      navigate('/assessment')
    }
  }, [navigate])

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }))
  }

  if (!result) {
    return (
      <div className="pt-20 min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  const getRiskColor = (level) => {
    const colors = {
      'Low': 'green',
      'Medium': 'yellow',
      'High': 'orange',
      'Very High': 'red'
    }
    return colors[level] || 'gray'
  }

  const getRiskIcon = () => {
    if (result.prediction === 0) {
      return <CheckCircle className="w-20 h-20 text-green-500" />
    }
    return <AlertTriangle className="w-20 h-20 text-red-500" />
  }

  const probability = (result.probability * 100).toFixed(1)
  const riskColor = getRiskColor(result.risk_level)
  const insights = result.ai_insights || {}

  return (
    <div className="pt-20 min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-4">
            Comprehensive Health Assessment
          </h1>
          <p className="text-xl text-gray-600">
            AI-powered cardiovascular risk analysis with personalized insights
          </p>
        </motion.div>

        {/* Main Result Card */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className={`card mb-8 border-2 ${
            result.prediction === 1 ? 'border-red-300 bg-red-50' : 'border-green-300 bg-green-50'
          }`}
        >
          <div className="text-center">
            <div className="flex justify-center mb-4">
              <PulsingHeart 
                size="xl" 
                color={result.prediction === 1 ? 'red' : 'green'} 
              />
            </div>
            <h2 className="text-3xl font-bold mb-2">
              {result.prediction === 1 ? 'Elevated Risk Detected' : 'Low Risk Profile'}
            </h2>
            <p className="text-lg text-gray-700 mb-6">
              {result.model_used}
            </p>
          </div>
        </motion.div>

        {/* Risk Details Grid */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {/* Risk Level with Progress Ring */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="card"
          >
            <h3 className="text-xl font-semibold mb-4 flex items-center">
              <TrendingUp className="w-6 h-6 mr-2 text-primary-600" />
              Risk Level
            </h3>
            <div className="flex flex-col items-center">
              <ProgressRing 
                progress={parseFloat(probability)} 
                size={160}
                strokeWidth={12}
                color={
                  riskColor === 'green' ? '#10b981' :
                  riskColor === 'yellow' ? '#f59e0b' :
                  riskColor === 'orange' ? '#f97316' :
                  '#ef4444'
                }
              >
                <div className="text-center">
                  <div className={`text-4xl font-bold text-${riskColor}-600`}>
                    {probability}%
                  </div>
                  <div className="text-sm text-gray-600">Risk</div>
                </div>
              </ProgressRing>
              <div className={`text-3xl font-bold text-${riskColor}-600 mt-4`}>
                {result.risk_level}
              </div>
            </div>
          </motion.div>

          {/* Risk Score */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="card"
          >
            <h3 className="text-xl font-semibold mb-4 flex items-center">
              <Activity className="w-6 h-6 mr-2 text-primary-600" />
              Risk Score
            </h3>
            <div className="flex flex-col items-center justify-center h-full">
              <motion.div 
                className="text-7xl font-bold text-primary-600 mb-2"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.6, type: "spring", stiffness: 200 }}
              >
                <AnimatedCounter end={result.risk_score} duration={2} />
              </motion.div>
              <p className="text-gray-600 text-center">
                Data-driven score based on clinical features
              </p>
            </div>
          </motion.div>
        </div>

        {/* Executive Summary */}
        {insights.executive_summary && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="card mb-8"
          >
            <button
              onClick={() => toggleSection('summary')}
              className="w-full flex items-center justify-between text-left"
            >
              <h3 className="text-2xl font-semibold flex items-center">
                <FileText className="w-6 h-6 mr-2 text-primary-600" />
                Executive Summary
              </h3>
              {expandedSections.summary ? <ChevronUp /> : <ChevronDown />}
            </button>
            
            {expandedSections.summary && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="mt-4"
              >
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                  <p className="text-lg text-gray-800 leading-relaxed">
                    {insights.executive_summary}
                  </p>
                </div>
              </motion.div>
            )}
          </motion.div>
        )}

        {/* Risk Analysis */}
        {insights.risk_analysis && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="card mb-8"
          >
            <h3 className="text-2xl font-semibold mb-4 flex items-center">
              <Heart className="w-6 h-6 mr-2 text-red-500" />
              Detailed Risk Analysis
            </h3>
            
            {/* Primary Risk Factors */}
            {insights.risk_analysis.primary_risk_factors && insights.risk_analysis.primary_risk_factors.length > 0 && (
              <div className="mb-6">
                <h4 className="font-semibold text-lg mb-3 text-red-600 flex items-center">
                  <AlertTriangle className="w-5 h-5 mr-2" />
                  Primary Risk Factors
                </h4>
                <div className="space-y-4">
                  {insights.risk_analysis.primary_risk_factors.map((factor, index) => (
                    <motion.div 
                      key={index} 
                      className="bg-red-50 border border-red-200 rounded-lg p-4"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      whileHover={{ scale: 1.02, boxShadow: "0 4px 12px rgba(239, 68, 68, 0.2)" }}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <h5 className="font-semibold text-red-900">{factor.factor}</h5>
                        <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                          factor.impact === 'Very High' ? 'bg-red-600 text-white' :
                          factor.impact === 'High' ? 'bg-orange-500 text-white' :
                          'bg-yellow-500 text-white'
                        }`}>
                          {factor.impact} Impact
                        </span>
                      </div>
                      <p className="text-sm text-gray-700 mb-2"><strong>Value:</strong> {factor.value}</p>
                      <p className="text-sm text-gray-700">{factor.explanation}</p>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}

            {/* Protective Factors */}
            {insights.risk_analysis.protective_factors && insights.risk_analysis.protective_factors.length > 0 && (
              <div className="mb-6">
                <h4 className="font-semibold text-lg mb-3 text-green-600 flex items-center">
                  <CheckCircle className="w-5 h-5 mr-2" />
                  Protective Factors
                </h4>
                <div className="grid md:grid-cols-2 gap-4">
                  {insights.risk_analysis.protective_factors.map((factor, index) => (
                    <motion.div 
                      key={index} 
                      className="bg-green-50 border border-green-200 rounded-lg p-4"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: index * 0.1 }}
                      whileHover={{ scale: 1.05 }}
                    >
                      <h5 className="font-semibold text-green-900 mb-2">{factor.factor}</h5>
                      <p className="text-sm text-gray-700">{factor.benefit}</p>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}

            {/* Modifiable Factors */}
            {insights.risk_analysis.modifiable_factors && insights.risk_analysis.modifiable_factors.length > 0 && (
              <div>
                <h4 className="font-semibold text-lg mb-3 text-blue-600">Modifiable Risk Factors</h4>
                <div className="flex flex-wrap gap-2">
                  {insights.risk_analysis.modifiable_factors.map((factor, index) => (
                    <span
                      key={index}
                      className="px-4 py-2 bg-blue-100 text-blue-700 rounded-full text-sm font-medium"
                    >
                      {factor}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* Lifestyle Recommendations */}
        {insights.lifestyle_recommendations && insights.lifestyle_recommendations.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="card mb-8"
          >
            <button
              onClick={() => toggleSection('lifestyle')}
              className="w-full flex items-center justify-between text-left mb-4"
            >
              <h3 className="text-2xl font-semibold flex items-center">
                <Lightbulb className="w-6 h-6 mr-2 text-yellow-500" />
                Personalized Lifestyle Recommendations
              </h3>
              {expandedSections.lifestyle ? <ChevronUp /> : <ChevronDown />}
            </button>
            
            {expandedSections.lifestyle && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="space-y-6"
              >
                {insights.lifestyle_recommendations.map((rec, index) => (
                  <motion.div 
                    key={index} 
                    className="bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200 rounded-lg p-6"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    whileHover={{ scale: 1.02, boxShadow: "0 8px 16px rgba(147, 51, 234, 0.15)" }}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <h4 className="font-bold text-xl text-purple-900">{rec.category}</h4>
                      <motion.span 
                        className={`px-3 py-1 rounded-full text-xs font-semibold ${
                          rec.priority === 'Critical' ? 'bg-red-600 text-white' :
                          rec.priority === 'High' ? 'bg-orange-500 text-white' :
                          'bg-yellow-500 text-white'
                        }`}
                        whileHover={{ scale: 1.1 }}
                      >
                        {rec.priority} Priority
                      </motion.span>
                    </div>
                    <p className="text-lg font-semibold text-purple-800 mb-2">{rec.recommendation}</p>
                    <p className="text-gray-700 mb-4">{rec.details}</p>
                    
                    <div className="bg-white rounded-lg p-4 mb-4">
                      <h5 className="font-semibold mb-2">Action Steps:</h5>
                      <ul className="list-disc list-inside space-y-1">
                        {rec.action_steps.map((step, idx) => (
                          <li key={idx} className="text-sm text-gray-700">{step}</li>
                        ))}
                      </ul>
                    </div>
                    
                    <div className="bg-green-100 border border-green-300 rounded-lg p-3">
                      <p className="text-sm font-semibold text-green-900">
                        <Target className="w-4 h-4 inline mr-1" />
                        Expected Benefit: {rec.expected_benefit}
                      </p>
                    </div>
                  </motion.div>
                ))}
              </motion.div>
            )}
          </motion.div>
        )}

        {/* Medical Considerations */}
        {insights.medical_considerations && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            className="card mb-8"
          >
            <button
              onClick={() => toggleSection('medical')}
              className="w-full flex items-center justify-between text-left mb-4"
            >
              <h3 className="text-2xl font-semibold flex items-center">
                <Activity className="w-6 h-6 mr-2 text-red-500" />
                Medical Considerations
              </h3>
              {expandedSections.medical ? <ChevronUp /> : <ChevronDown />}
            </button>
            
            {expandedSections.medical && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="space-y-6"
              >
                {/* Immediate Actions */}
                {insights.medical_considerations.immediate_actions && insights.medical_considerations.immediate_actions.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-lg mb-3 text-red-600">Immediate Actions Required</h4>
                    <div className="space-y-3">
                      {insights.medical_considerations.immediate_actions.map((action, index) => (
                        <div key={index} className="bg-red-50 border-l-4 border-red-500 p-4">
                          <p className="font-semibold text-red-900">{action.action}</p>
                          <p className="text-sm text-gray-700 mt-1">{action.reason}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Screening Recommendations */}
                {insights.medical_considerations.screening_recommendations && insights.medical_considerations.screening_recommendations.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-lg mb-3 text-blue-600">Recommended Screenings</h4>
                    <div className="grid md:grid-cols-2 gap-4">
                      {insights.medical_considerations.screening_recommendations.map((screening, index) => (
                        <div key={index} className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                          <h5 className="font-semibold text-blue-900">{screening.test}</h5>
                          <p className="text-sm text-gray-700 mt-1">
                            <Calendar className="w-4 h-4 inline mr-1" />
                            {screening.frequency}
                          </p>
                          <p className="text-sm text-gray-600 mt-2">{screening.purpose}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </motion.div>
        )}

        {/* Follow-up Recommendations */}
        {insights.follow_up_recommendations && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9 }}
            className="card mb-8"
          >
            <h3 className="text-2xl font-semibold mb-4 flex items-center">
              <Calendar className="w-6 h-6 mr-2 text-primary-600" />
              Follow-up Plan
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <motion.div 
                className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-6"
                whileHover={{ scale: 1.05, boxShadow: "0 8px 20px rgba(37, 99, 235, 0.2)" }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <h4 className="font-semibold mb-2">Next Assessment</h4>
                <p className="text-3xl font-bold text-primary-600">{insights.follow_up_recommendations.next_assessment}</p>
              </motion.div>
              <motion.div 
                className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-6"
                whileHover={{ scale: 1.05, boxShadow: "0 8px 20px rgba(147, 51, 234, 0.2)" }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <h4 className="font-semibold mb-2">Monitoring Frequency</h4>
                <p className="text-3xl font-bold text-purple-600">{insights.follow_up_recommendations.monitoring_frequency}</p>
              </motion.div>
            </div>
          </motion.div>
        )}

        {/* Disclaimer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.0 }}
          className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 mb-8"
        >
          <div className="flex items-start space-x-3">
            <AlertTriangle className="w-6 h-6 text-yellow-600 flex-shrink-0" />
            <div>
              <h3 className="font-semibold text-yellow-900 mb-2">Important Medical Disclaimer</h3>
              <p className="text-yellow-800 text-sm leading-relaxed">
                This assessment is for research and clinical decision support only. It is not a substitute 
                for professional medical diagnosis, treatment, or advice. Please consult with a qualified 
                healthcare provider to discuss these results and determine appropriate next steps. If you 
                are experiencing chest pain, shortness of breath, or other emergency symptoms, call 911 immediately.
              </p>
            </div>
          </div>
        </motion.div>

        {/* Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.1 }}
          className="flex flex-wrap gap-4 justify-center"
        >
          <motion.button
            onClick={() => {
              localStorage.removeItem('assessmentResult')
              navigate('/assessment')
            }}
            className="btn-primary flex items-center space-x-2"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <RotateCcw className="w-5 h-5" />
            <span>New Assessment</span>
          </motion.button>
          
          <motion.button
            onClick={() => window.print()}
            className="btn-secondary flex items-center space-x-2"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Download className="w-5 h-5" />
            <span>Download Report</span>
          </motion.button>
        </motion.div>
      </div>
    </div>
  )
}

export default ResultsPage
