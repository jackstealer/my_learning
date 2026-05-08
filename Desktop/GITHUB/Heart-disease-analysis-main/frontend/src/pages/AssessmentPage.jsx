import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { useForm } from 'react-hook-form'
import axios from 'axios'
import { 
  User, 
  Activity, 
  Heart, 
  Cigarette, 
  Wine,
  Moon,
  AlertCircle,
  Loader2,
  CheckCircle2
} from 'lucide-react'
import GlowingCard from '../components/GlowingCard'
import PulsingHeart from '../components/PulsingHeart'

const AssessmentPage = () => {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [completedSections, setCompletedSections] = useState(new Set())
  const navigate = useNavigate()
  const { register, handleSubmit, formState: { errors }, watch } = useForm()

  const onSubmit = async (data) => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await axios.post('/api/predict', data)
      // Store result and navigate
      localStorage.setItem('assessmentResult', JSON.stringify(response.data))
      navigate('/results')
    } catch (err) {
      setError(err.response?.data?.message || 'Assessment failed. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const formSections = [
    {
      title: 'Personal Information',
      icon: <User className="w-6 h-6" />,
      fields: [
        {
          name: 'Sex',
          label: 'Sex',
          type: 'select',
          options: ['Male', 'Female'],
          required: true
        },
        {
          name: 'AgeCategory',
          label: 'Age Category',
          type: 'select',
          options: ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'],
          required: true
        },
        {
          name: 'Race',
          label: 'Race/Ethnicity',
          type: 'select',
          options: ['White', 'Black', 'Asian', 'Hispanic', 'American Indian/Alaskan Native', 'Other'],
          required: true
        }
      ]
    },
    {
      title: 'Physical Health',
      icon: <Activity className="w-6 h-6" />,
      fields: [
        {
          name: 'BMI',
          label: 'BMI (Body Mass Index)',
          type: 'number',
          step: '0.01',
          min: '10',
          max: '100',
          placeholder: 'e.g., 25.5',
          required: true,
          hint: 'Normal range: 18.5-24.9'
        },
        {
          name: 'PhysicalHealth',
          label: 'Physical Health (poor days in last 30)',
          type: 'number',
          min: '0',
          max: '30',
          placeholder: '0-30',
          required: true
        },
        {
          name: 'MentalHealth',
          label: 'Mental Health (poor days in last 30)',
          type: 'number',
          min: '0',
          max: '30',
          placeholder: '0-30',
          required: true
        },
        {
          name: 'SleepTime',
          label: 'Sleep Time (hours per night)',
          type: 'number',
          min: '0',
          max: '24',
          placeholder: 'e.g., 7',
          required: true,
          hint: 'Recommended: 7-9 hours'
        },
        {
          name: 'GenHealth',
          label: 'General Health',
          type: 'select',
          options: ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'],
          required: true
        }
      ]
    },
    {
      title: 'Lifestyle Factors',
      icon: <Cigarette className="w-6 h-6" />,
      fields: [
        {
          name: 'Smoking',
          label: 'Do you smoke?',
          type: 'select',
          options: ['Yes', 'No'],
          required: true
        },
        {
          name: 'AlcoholDrinking',
          label: 'Heavy alcohol drinking?',
          type: 'select',
          options: ['Yes', 'No'],
          required: true
        },
        {
          name: 'PhysicalActivity',
          label: 'Physical activity in past 30 days?',
          type: 'select',
          options: ['Yes', 'No'],
          required: true
        }
      ]
    },
    {
      title: 'Medical History',
      icon: <Heart className="w-6 h-6" />,
      fields: [
        {
          name: 'Stroke',
          label: 'History of stroke?',
          type: 'select',
          options: ['Yes', 'No'],
          required: true
        },
        {
          name: 'DiffWalking',
          label: 'Difficulty walking or climbing stairs?',
          type: 'select',
          options: ['Yes', 'No'],
          required: true
        },
        {
          name: 'Diabetic',
          label: 'Diabetic status',
          type: 'select',
          options: ['Yes', 'No', 'No, borderline diabetes', 'Yes (during pregnancy)'],
          required: true
        },
        {
          name: 'Asthma',
          label: 'History of asthma?',
          type: 'select',
          options: ['Yes', 'No'],
          required: true
        },
        {
          name: 'KidneyDisease',
          label: 'History of kidney disease?',
          type: 'select',
          options: ['Yes', 'No'],
          required: true
        },
        {
          name: 'SkinCancer',
          label: 'History of skin cancer?',
          type: 'select',
          options: ['Yes', 'No'],
          required: true
        }
      ]
    }
  ]

  return (
    <div className="pt-20 min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="flex justify-center mb-4">
            <PulsingHeart size="xl" color="red" />
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-4">
            Cardiovascular Risk Assessment
          </h1>
          <p className="text-xl text-gray-600">
            Complete the form below for AI-powered risk analysis
          </p>
        </motion.div>

        {/* Error Alert */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-start space-x-3"
          >
            <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-semibold text-red-900">Assessment Error</h3>
              <p className="text-red-700">{error}</p>
            </div>
          </motion.div>
        )}

        {/* Form */}
        <motion.form
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          onSubmit={handleSubmit(onSubmit)}
          className="space-y-8"
        >
          {formSections.map((section, sectionIndex) => (
            <GlowingCard
              key={sectionIndex}
              className="card"
              glowColor="primary"
            >
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: sectionIndex * 0.1 }}
              >
                <div className="flex items-center justify-between mb-6 pb-4 border-b">
                  <div className="flex items-center space-x-3">
                    <motion.div 
                      className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center text-primary-600"
                      whileHover={{ rotate: 360 }}
                      transition={{ duration: 0.5 }}
                    >
                      {section.icon}
                    </motion.div>
                    <h2 className="text-2xl font-semibold">{section.title}</h2>
                  </div>
                  {completedSections.has(sectionIndex) && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: "spring", stiffness: 500 }}
                    >
                      <CheckCircle2 className="w-6 h-6 text-green-500" />
                    </motion.div>
                  )}
                </div>

              <div className="grid md:grid-cols-2 gap-6">
                {section.fields.map((field, fieldIndex) => (
                  <div key={fieldIndex} className={field.type === 'select' && field.options.length > 10 ? 'md:col-span-2' : ''}>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      {field.label}
                      {field.required && <span className="text-red-500 ml-1">*</span>}
                    </label>
                    
                    {field.type === 'select' ? (
                      <motion.select
                        {...register(field.name, { required: field.required })}
                        className="input-field"
                        whileFocus={{ scale: 1.02 }}
                        transition={{ type: "spring", stiffness: 300 }}
                      >
                        <option value="">Select...</option>
                        {field.options.map((option, i) => (
                          <option key={i} value={option}>{option}</option>
                        ))}
                      </motion.select>
                    ) : (
                      <motion.input
                        type={field.type}
                        step={field.step}
                        min={field.min}
                        max={field.max}
                        placeholder={field.placeholder}
                        {...register(field.name, { required: field.required })}
                        className="input-field"
                        whileFocus={{ scale: 1.02 }}
                        transition={{ type: "spring", stiffness: 300 }}
                      />
                    )}
                    
                    {field.hint && (
                      <p className="text-sm text-gray-500 mt-1">{field.hint}</p>
                    )}
                    
                    {errors[field.name] && (
                      <motion.p 
                        className="text-sm text-red-600 mt-1"
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                      >
                        This field is required
                      </motion.p>
                    )}
                  </div>
                ))}
              </div>
              </motion.div>
            </GlowingCard>
          ))}

          {/* Disclaimer */}
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
            <div className="flex items-start space-x-3">
              <AlertCircle className="w-6 h-6 text-yellow-600 flex-shrink-0" />
              <div>
                <h3 className="font-semibold text-yellow-900 mb-2">Medical Disclaimer</h3>
                <p className="text-yellow-800 text-sm leading-relaxed">
                  This assessment is for research and clinical decision support only. It is not a substitute 
                  for professional medical diagnosis, treatment, or advice. Always consult with qualified 
                  healthcare providers for medical decisions.
                </p>
              </div>
            </div>
          </div>

          {/* Submit Button */}
          <div className="flex justify-center">
            <motion.button
              type="submit"
              disabled={isLoading}
              className="btn-primary text-lg px-12 py-4 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
              whileHover={{ scale: isLoading ? 1 : 1.05 }}
              whileTap={{ scale: isLoading ? 1 : 0.95 }}
            >
              <AnimatePresence mode="wait">
                {isLoading ? (
                  <motion.div
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex items-center space-x-2"
                  >
                    <Loader2 className="w-6 h-6 animate-spin" />
                    <span>Analyzing...</span>
                  </motion.div>
                ) : (
                  <motion.div
                    key="ready"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex items-center space-x-2"
                  >
                    <Activity className="w-6 h-6" />
                    <span>Analyze Risk</span>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.button>
          </div>
        </motion.form>
      </div>
    </div>
  )
}

export default AssessmentPage
