<template>
  <div id="app">
    <h1>Spectrum Viewer</h1>
    <div class="chart-container">
      <canvas ref="spectrumChart"></canvas>
    </div>
    <div class="controls">
      <button @click="resetZoom">Reset Zoom</button>
      <!-- <button @click="generateRandomSpectrum">Generate Random Spectrum</button> -->
      <label for="fileInput" class="file-input-label">
        <input
          id="fileInput"
          type="file"
          accept=".csv,.txt"
          @change="loadSpectrumFromFile"
          style="display: none;"
        />
        Load Spectrum from File
      </label>
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

// Store current spectrum data for saving/processing
const currentSpectrum = ref<{
  wavelengths: number[]
  intensities: number[]
  filename?: string
  timestamp: Date
} | null>(null)

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
        borderWidth: 1,
        fill: false,
        tension: 0,
        pointRadius: 0,
        pointHoverRadius: 0,
        pointHitRadius: 0,
        pointHoverBorderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
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
            maxTicksLimit: 20,
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
  
  // Add zoom event listener for dynamic downsampling
  chart.canvas.addEventListener('wheel', () => {
    setTimeout(handleZoom, 100) // Delay to allow zoom to complete
  })
  
  // Listen for zoom plugin events
  chart.canvas.addEventListener('mousedown', () => {
    setTimeout(handleZoom, 100) // Delay to allow zoom to complete
  })
}

const resetZoom = () => {
  if (chart) {
    chart.resetZoom()
    // Restore full dataset when zoom is reset using downsampleDataForZoom
    if (fullResolutionData.value) {
      // Get the full wavelength range for reset
      const fullRange = {
        min: Math.min(...fullResolutionData.value.wavelengths),
        max: Math.max(...fullResolutionData.value.wavelengths)
      }
      
      const { wavelengths: finalWavelengths, intensities: finalIntensities } = downsampleDataForZoom(
        fullResolutionData.value.wavelengths,
        fullResolutionData.value.intensities,
        fullRange,
        2000 // Use standard downsampling for full view
      )
      
      chart.data.labels = finalWavelengths
      chart.data.datasets[0].data = finalIntensities
      chart.update('none')
    }
  }
}

const handleZoom = () => {
  if (!chart || !fullResolutionData.value) return
  
      try {
      const xAxis = chart.scales.x
      
      if (xAxis && xAxis.min !== undefined && xAxis.max !== undefined) {
        const zoomRange = { min: xAxis.min, max: xAxis.max }
        
        // Get data for current zoom level
        const { wavelengths: zoomWavelengths, intensities: zoomIntensities } = downsampleDataForZoom(
          fullResolutionData.value.wavelengths,
          fullResolutionData.value.intensities,
          zoomRange,
          2000 // Use fewer points for zoomed views
        )
        
        // Update chart with zoom-optimized data
        chart.data.labels = zoomWavelengths
        chart.data.datasets[0].data = zoomIntensities
        chart.update('none')
        
        console.log(`Zoom range: ${zoomRange.min.toFixed(1)} - ${zoomRange.max.toFixed(1)} nm, Data points: ${zoomWavelengths.length}`)
      }
    } catch (error) {
      console.error('Error handling zoom:', error)
    }
}

const loadSpectrumFromFile = (event: Event) => {
  const fileInput = event.target as HTMLInputElement
  if (!fileInput.files || fileInput.files.length === 0) {
    alert('Please select a file to load.')
    return
  }

  const file = fileInput.files[0]
  const reader = new FileReader()

  reader.onload = (e) => {
    if (e.target?.result) {
      try {
        const data = parseCSV(e.target.result as string)
        if (data && data.length > 0) {
          const wavelengths = data.map(row => Number(row[0]))
          const intensities = data.map(row => Number(row[1]))

          // Store full resolution data for zoom operations
          fullResolutionData.value = {
            wavelengths: wavelengths,
            intensities: intensities
          }
          
          // Store current spectrum data for saving/processing
          currentSpectrum.value = {
              wavelengths: wavelengths,
              intensities: intensities,
              filename: file.name,
              timestamp: new Date()
            }
          
          console.log(currentSpectrum.value)
          // Show processing info for large datasets
          if (data.length > 2000) {
            console.log(`Processing large dataset: ${data.length} points, downsampling to 2000 points for smooth rendering`)
          }
          
          // Downsample data if it's too large for smooth rendering
          // const { wavelengths: finalWavelengths, intensities: finalIntensities } = downsampleData(wavelengths, intensities)
          const fullRange = {
            min: Math.min(...fullResolutionData.value.wavelengths),
            max: Math.max(...fullResolutionData.value.wavelengths)
          }
      
          const { wavelengths: finalWavelengths, intensities: finalIntensities } = downsampleDataForZoom(
            fullResolutionData.value.wavelengths,
            fullResolutionData.value.intensities,
            fullRange,
            2000 // Use standard downsampling for full view
          )
          if (chart) {
            chart.data.labels = finalWavelengths
            chart.data.datasets[0].data = finalIntensities
            chart.update('none') // Use 'none' mode for faster updates
            chart.resetZoom()
            
            
          } else {
            alert('Chart not initialized. Please ensure the canvas is visible.')
          }
        } else {
          alert('No valid data found in the selected file.')
        }
      } catch (error) {
        alert(`Error loading spectrum from file: ${error}`)
      }
    }
  }

  reader.onerror = () => {
    alert('Error reading file.')
  }

  reader.readAsText(file)
}

const parseCSV = (csvString: string) => {
  const lines = csvString.trim().split('\n')
  const data: string[][] = []

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim()
    if (line && !line.startsWith('#')) { // Skip empty lines and comments
      const row = line.split(',').map(cell => cell.trim())
      if (row.length >= 2) { // Ensure we have at least 2 columns
        data.push(row)
      }
    }
  }
  return data
}

const downsampleDataForZoom = (wavelengths: number[], intensities: number[], zoomRange: { min: number, max: number }, maxPoints: number = 2000) => {
  // Find indices for the zoom range
  const startIndex = wavelengths.findIndex(w => w >= zoomRange.min)
  const endIndex = wavelengths.findIndex(w => w > zoomRange.max)
  
  if (startIndex === -1 || endIndex === -1) {
    return { wavelengths, intensities }
  }
  
  const visibleWavelengths = wavelengths.slice(startIndex, endIndex)
  const visibleIntensities = intensities.slice(startIndex, endIndex)
  
  // If visible data is small enough, return as is
  if (visibleWavelengths.length <= maxPoints) {
    return { wavelengths: visibleWavelengths, intensities: visibleIntensities }
  }
  
  // Downsample visible data
  const step = Math.ceil(visibleWavelengths.length / maxPoints)
  const downsampledWavelengths: number[] = []
  const downsampledIntensities: number[] = []
  
  for (let i = 0; i < visibleWavelengths.length; i += step) {
    downsampledWavelengths.push(visibleWavelengths[i])
    downsampledIntensities.push(visibleIntensities[i])
  }
  
  return { wavelengths: downsampledWavelengths, intensities: downsampledIntensities }
}

// Store original full resolution data
const fullResolutionData = ref<{
  wavelengths: number[]
  intensities: number[]
} | null>(null)

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

.file-input-label {
  background-color: #4CAF50;
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

.file-input-label:hover {
  background-color: #45a049;
}

.file-input-label:active {
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
