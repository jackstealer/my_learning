import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import axios from 'axios'
import { 
  TrendingUp, 
  Users, 
  Activity, 
  Database,
  BarChart3,
  PieChart,
  Loader2
} from 'lucide-react'
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend,
  ResponsiveContainer,
  PieChart as RePieChart,
  Pie,
  Cell
} from 'recharts'

const DashboardPage = () => {
  const [stats, setStats] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsRes, modelRes] = await Promise.all([
          axios.get('/api/stats'),
          axios.get('/api/model-info')
        ])
        setStats(statsRes.data)
        setModelInfo(modelRes.data)
      } catch (error) {
        console.error('Error fetching data:', error)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  if (loading) {
    return (
      <div className="pt-20 min-h-screen flex items-center justify-center">
        <Loader2 className="w-12 h-12 animate-spin text-primary-600" />
      </div>
    )
  }

  const statCards = [
    {
      icon: <Users className="w-8 h-8" />,
      label: 'Total Patients',
      value: stats?.total_patients || '4,500',
      color: 'blue'
    },
    {
      icon: <Activity className="w-8 h-8" />,
      label: 'Disease Prevalence',
      value: `${stats?.disease_prevalence || '10.7'}%`,
      color: 'red'
    },
    {
      icon: <TrendingUp className="w-8 h-8" />,
      label: 'Average Age',
      value: stats?.avg_age || '54.4',
      color: 'green'
    },
    {
      icon: <Database className="w-8 h-8" />,
      label: 'High Risk Patients',
      value: stats?.high_risk_patients || '483',
      color: 'orange'
    }
  ]

  // Feature importance data
  const featureData = modelInfo?.feature_stats?.feature_importance 
    ? Object.entries(modelInfo.feature_stats.feature_importance)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([name, value]) => ({
          name,
          importance: (value * 100).toFixed(1)
        }))
    : []

  const COLORS = ['#2563eb', '#0891b2', '#059669', '#d97706', '#dc2626']

  return (
    <div className="pt-20 min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-4">
            Analytics Dashboard
          </h1>
          <p className="text-xl text-gray-600">
            Real-time insights and model performance metrics
          </p>
        </motion.div>

        {/* Stats Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {statCards.map((stat, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="card"
            >
              <div className={`w-16 h-16 bg-${stat.color}-100 rounded-xl flex items-center justify-center text-${stat.color}-600 mb-4`}>
                {stat.icon}
              </div>
              <div className="text-3xl font-bold text-slate-900 mb-1">
                {stat.value}
              </div>
              <div className="text-gray-600">{stat.label}</div>
            </motion.div>
          ))}
        </div>

        {/* Charts */}
        <div className="grid lg:grid-cols-2 gap-8 mb-12">
          {/* Feature Importance Chart */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="card"
          >
            <h2 className="text-2xl font-semibold mb-6 flex items-center">
              <BarChart3 className="w-6 h-6 mr-2 text-primary-600" />
              Top Risk Factors
            </h2>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={featureData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="name" 
                  angle={-45} 
                  textAnchor="end" 
                  height={100}
                  fontSize={12}
                />
                <YAxis />
                <Tooltip />
                <Bar dataKey="importance" fill="#2563eb" />
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Model Info */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 }}
            className="card"
          >
            <h2 className="text-2xl font-semibold mb-6 flex items-center">
              <Activity className="w-6 h-6 mr-2 text-primary-600" />
              Model Information
            </h2>
            <div className="space-y-4">
              <div className="flex justify-between items-center p-4 bg-blue-50 rounded-lg">
                <span className="font-medium">Model Type</span>
                <span className="text-primary-600 font-semibold">
                  {modelInfo?.model_type || 'Data-Driven AI'}
                </span>
              </div>
              <div className="flex justify-between items-center p-4 bg-green-50 rounded-lg">
                <span className="font-medium">Status</span>
                <span className="text-green-600 font-semibold">
                  {modelInfo?.loaded ? 'Active' : 'Inactive'}
                </span>
              </div>
              <div className="flex justify-between items-center p-4 bg-purple-50 rounded-lg">
                <span className="font-medium">Features</span>
                <span className="text-purple-600 font-semibold">
                  {modelInfo?.features?.length || 17}
                </span>
              </div>
              <div className="flex justify-between items-center p-4 bg-orange-50 rounded-lg">
                <span className="font-medium">Framework</span>
                <span className="text-orange-600 font-semibold">
                  {modelInfo?.framework || 'Statistical Analysis'}
                </span>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Dataset Info */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="card"
        >
          <h2 className="text-2xl font-semibold mb-6 flex items-center">
            <Database className="w-6 h-6 mr-2 text-primary-600" />
            Dataset Information
          </h2>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center p-6 bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl">
              <div className="text-4xl font-bold text-blue-600 mb-2">
                {stats?.total_patients || '4,500'}
              </div>
              <div className="text-gray-700 font-medium">Total Records</div>
            </div>
            <div className="text-center p-6 bg-gradient-to-br from-green-50 to-green-100 rounded-xl">
              <div className="text-4xl font-bold text-green-600 mb-2">
                {modelInfo?.features?.length || 17}
              </div>
              <div className="text-gray-700 font-medium">Clinical Features</div>
            </div>
            <div className="text-center p-6 bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl">
              <div className="text-4xl font-bold text-purple-600 mb-2">
                100%
              </div>
              <div className="text-gray-700 font-medium">Data-Driven</div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default DashboardPage
