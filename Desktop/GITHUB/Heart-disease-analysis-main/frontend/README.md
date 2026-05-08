# CardioAI - Professional Heart Disease Risk Assessment Platform

Modern, interactive React + Three.js frontend for the CardioAI heart disease prediction system.

## 🚀 Features

- **3D Animations**: Interactive 3D heart visualization using Three.js
- **Modern UI**: Built with React, Tailwind CSS, and Framer Motion
- **Responsive Design**: Mobile-first approach with beautiful animations
- **Real-time Assessment**: Instant cardiovascular risk analysis
- **Interactive Dashboard**: Data visualization with Recharts
- **Professional Design**: Healthcare-grade UI/UX

## 📦 Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool
- **Three.js** - 3D graphics
- **@react-three/fiber** - React renderer for Three.js
- **@react-three/drei** - Three.js helpers
- **Framer Motion** - Animations
- **Tailwind CSS** - Styling
- **React Router** - Navigation
- **Axios** - HTTP client
- **Recharts** - Data visualization
- **React Hook Form** - Form management
- **Lucide React** - Icons

## 🛠️ Installation

1. **Install dependencies:**

```bash
cd frontend
npm install
```

2. **Start development server:**

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

3. **Build for production:**

```bash
npm run build
```

## 🔧 Configuration

### API Proxy

The Vite config is set up to proxy API requests to the Flask backend:

```javascript
// vite.config.js
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:5000',
      changeOrigin: true,
    }
  }
}
```

### Environment Variables

Create a `.env` file in the frontend directory:

```env
VITE_API_URL=http://localhost:5000
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Navbar.jsx          # Navigation bar
│   │   ├── Footer.jsx          # Footer component
│   │   └── Heart3D.jsx         # 3D heart animation
│   ├── pages/
│   │   ├── HomePage.jsx        # Landing page with 3D hero
│   │   ├── AssessmentPage.jsx  # Risk assessment form
│   │   ├── ResultsPage.jsx     # Assessment results
│   │   ├── DashboardPage.jsx   # Analytics dashboard
│   │   └── AboutPage.jsx       # About page
│   ├── App.jsx                 # Main app component
│   ├── main.jsx                # Entry point
│   └── index.css               # Global styles
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
└── postcss.config.js
```

## 🎨 Customization

### Colors

Edit `tailwind.config.js` to customize the color scheme:

```javascript
theme: {
  extend: {
    colors: {
      primary: {
        600: '#2563eb',  // Main brand color
        // ... other shades
      }
    }
  }
}
```

### 3D Heart Animation

Customize the 3D heart in `src/components/Heart3D.jsx`:

```javascript
<MeshDistortMaterial
  color="#2563eb" // Heart color
  distort={0.3} // Distortion amount
  speed={2} // Animation speed
/>
```

## 🔌 API Integration

The frontend connects to the Flask backend API:

### Endpoints Used

- `POST /api/predict` - Submit assessment data
- `GET /api/model-info` - Get model information
- `GET /api/stats` - Get dashboard statistics

### Example API Call

```javascript
import axios from "axios";

const assessmentData = {
  BMI: 28.5,
  Smoking: "No",
  // ... other fields
};

const response = await axios.post("/api/predict", assessmentData);
console.log(response.data);
```

## 🚀 Deployment

### Build for Production

```bash
npm run build
```

This creates an optimized build in the `dist/` directory.

### Deploy to Vercel

1. Install Vercel CLI:

```bash
npm i -g vercel
```

2. Deploy:

```bash
vercel
```

### Deploy to Netlify

1. Build the project:

```bash
npm run build
```

2. Deploy the `dist/` folder to Netlify

### Environment Variables for Production

Set these in your deployment platform:

```
VITE_API_URL=https://your-backend-api.com
```

## 📱 Responsive Design

The app is fully responsive with breakpoints:

- **Mobile**: < 768px
- **Tablet**: 768px - 1024px
- **Desktop**: > 1024px

## ⚡ Performance

- **Code Splitting**: Automatic route-based code splitting
- **Lazy Loading**: Components loaded on demand
- **Optimized Images**: WebP format with fallbacks
- **Minification**: CSS and JS minified in production
- **Tree Shaking**: Unused code removed

## 🧪 Testing

```bash
# Run tests (when configured)
npm test

# Run linter
npm run lint
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For support, email contact@cardioai.com or open an issue on GitHub.

## ⚠️ Medical Disclaimer

This platform is for research and clinical decision support only. It is not a substitute for professional medical diagnosis, treatment, or advice. Always consult with qualified healthcare providers for medical decisions.
