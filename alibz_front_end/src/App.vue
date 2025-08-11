<template>
  <div id="app">
    <h1>Spectrum Viewer</h1>
    <div class="chart-container">
      <canvas ref="spectrumChart"></canvas>
    </div>
    <div class="controls">
      <button @click="resetZoom">Reset Zoom</button>
      <button @click="generateRandomSpectrum">Generate Random Spectrum</button>
    </div>
    <div class="instructions">
      <p><strong>Zoom Controls:</strong></p>
      <ul>
        <li><strong>Mouse Wheel:</strong> Zoom in/out</li>
        <li><strong>Drag Selection:</strong> Click and drag to select a zoom area</li>
        <li><strong>Pan:</strong> Hold Ctrl + click and drag to move around when zoomed in</li>
        <li><strong>Double Click:</strong> Reset zoom to full view</li>
      </ul>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue'
import Chart from 'chart.js/auto'
import zoomPlugin from 'chartjs-plugin-zoom'

// Register the zoom plugin
Chart.register(zoomPlugin)

const spectrumChart = ref<HTMLCanvasElement>()
let chart: Chart | null = null

// Sample spectrum data (wavelength in nm, intensity in arbitrary units)
const generateSpectrumData = () => {
  const wavelengths = []
  const intensities = []
  
  // Generate wavelength range from 400nm to 700nm
  for (let i = 400; i <= 700; i += 1) {
    wavelengths.push(i)
    
    // Create a realistic spectrum with multiple peaks
    const peak1 = Math.exp(-Math.pow((i - 450) / 20, 2)) * 0.8
    const peak2 = Math.exp(-Math.pow((i - 550) / 25, 2)) * 1.0
    const peak3 = Math.exp(-Math.pow((i - 650) / 30, 2)) * 0.6
    const baseline = 0.1 + 0.05 * Math.sin(i * 0.02)
    const noise = (Math.random() - 0.5) * 0.02
    
    intensities.push(peak1 + peak2 + peak3 + baseline + noise)
  }
  
  return { wavelengths, intensities }
}

const createChart = () => {
  if (!spectrumChart.value) return
  
  const { wavelengths, intensities } = generateSpectrumData()
  
  const ctx = spectrumChart.value.getContext('2d')
  if (!ctx) return
  
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: wavelengths,
      datasets: [{
        label: 'Intensity',
        data: intensities,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0)',
        borderWidth: 2,
        fill: true,
        tension: 0.1,
        pointRadius: 0,
        pointHoverRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        intersect: false,
        mode: 'index'
      },
      plugins: {
        title: {
          display: true,
          text: 'Optical Spectrum',
          font: {
            size: 16,
            weight: 'bold'
          }
        },
        legend: {
          display: false
        },
        zoom: {
          pan: {
            enabled: true,
            mode: 'xy',
            modifierKey: 'ctrl'
          },
          zoom: {
            wheel: {
              enabled: true
            },
            pinch: {
              enabled: true
            },
            mode: 'xy',
            drag: {
              enabled: true,
              backgroundColor: 'rgba(75, 192, 192, 0.3)',
              borderColor: 'rgba(75, 192, 192, 0.8)',
              borderWidth: 1
            }
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          display: true,
          title: {
            display: true,
            text: 'Wavelength (nm)',
            font: {
              size: 14,
              weight: 'bold'
            }
          },
          ticks: {
            display: true,
            callback: function(value: string | number) {
              return Number(value) + ' nm'
            }
          },
          grid: {
            display: true,
            color: 'rgba(0, 0, 0, 0.1)'
          }
        },
        y: {
          type: 'linear',
          display: true,
          title: {
            display: true,
            text: 'Intensity (a. u.)',
            font: {
              size: 14,
              weight: 'bold'
            }
          },
          ticks: {
            display: true,
            callback: function(value: string | number) {
              return Number(value).toFixed(2)
            }
          },
          grid: {
            display: true,
            color: 'rgba(0, 0, 0, 0.1)'
          }
        }
      }
    }
  })
}

const resetZoom = () => {
  if (chart) {
    chart.resetZoom()
  }
}

const generateRandomSpectrum = () => {
  if (chart) {
    const { wavelengths, intensities } = generateSpectrumData()
    chart.data.labels = wavelengths
    chart.data.datasets[0].data = intensities
    chart.update()
    chart.resetZoom()
  }
}

onMounted(async () => {
  await nextTick()
  createChart()
})
</script>

<style scoped>
#app {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

h1 {
  text-align: center;
  color: #333;
  margin-bottom: 30px;
}

.chart-container {
  position: relative;
  height: 500px;
  margin-bottom: 20px;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 20px;
  background: white;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.controls {
  text-align: center;
  margin-top: 20px;
}

button {
  background-color: #4CAF50;
  border: none;
  color: white;
  padding: 12px 24px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
  margin: 4px 8px;
  cursor: pointer;
  border-radius: 4px;
  transition: background-color 0.3s;
}

button:hover {
  background-color: #45a049;
}

button:active {
  background-color: #3d8b40;
}

.instructions {
  margin-top: 30px;
  padding: 20px;
  background-color: #f8f9fa;
  border-radius: 8px;
  border-left: 4px solid #4CAF50;
}

.instructions p {
  margin: 0 0 15px 0;
  color: #333;
  font-size: 16px;
}

.instructions ul {
  margin: 0;
  padding-left: 20px;
}

.instructions li {
  margin: 8px 0;
  color: #555;
  line-height: 1.4;
}

.instructions strong {
  color: #333;
}
</style>
