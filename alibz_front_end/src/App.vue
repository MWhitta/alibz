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
    
    <!-- Analysis Controls -->
    <div class="analysis-controls">
      <h3>Analysis Parameters</h3>
      <div class="control-group">
        <label for="nSigma">N-Sigma:</label>
        <input
          id="nSigma"
          v-model.number="nSigma"
          type="number"
          min="1"
          max="10"
          step="1"
          class="number-input"
        />
      </div>
      <div class="control-group">
        <label for="subtractBackground">Subtract Background:</label>
        <input
          id="subtractBackground"
          v-model="subtractBackground"
          type="checkbox"
          class="checkbox-input"
        />
      </div>
      <div class="control-group">
        <label for="elements">Elements:</label>
        <div class="elements-selector">
          <input
            v-model="elementSearch"
            @input="filterElements"
            @focus="showElementDropdown = true"
            @blur="handleElementBlur"
            type="text"
            placeholder="Search elements..."
            class="element-search-input"
          />
          <div v-if="showElementDropdown" class="elements-dropdown">
            <div
              v-for="element in filteredElements"
              :key="element.symbol"
              @click="toggleElement(element.symbol)"
              :class="['element-option', { 'selected': selectedElements.includes(element.symbol) }]"
            >
              <span class="element-symbol">{{ element.symbol }}</span>
              <span class="element-name">{{ element.name }}</span>
            </div>
          </div>
          <div class="selected-elements">
            <span
              v-for="symbol in selectedElements"
              :key="symbol"
              class="selected-element-tag"
            >
              {{ symbol }}
              <button @click="removeElement(symbol)" class="remove-element-btn">&times;</button>
            </span>
          </div>
        </div>
      </div>
      <button 
        @click="performOneClickAnalysis" 
        :disabled="!currentSpectrum || selectedElements.length === 0"
        class="analysis-button"
      >
        One Click Analysis
      </button>
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
import { io, Socket } from 'socket.io-client'

// Register the zoom plugin
Chart.register(zoomPlugin)

const spectrumChart = ref<HTMLCanvasElement>()
let chart: Chart | null = null

// Socket.IO client
let socket: Socket | null = null

// Analysis parameters
const nSigma = ref(1)
const subtractBackground = ref(true)

// Element selection variables
const elementSearch = ref('')
const showElementDropdown = ref(false)
const selectedElements = ref<string[]>([])
const filteredElements = ref<Array<{symbol: string, name: string}>>([])

// Periodic table data
const periodicTable = [
  { symbol: 'H', name: 'Hydrogen' }, { symbol: 'He', name: 'Helium' },
  { symbol: 'Li', name: 'Lithium' }, { symbol: 'Be', name: 'Beryllium' },
  { symbol: 'B', name: 'Boron' }, { symbol: 'C', name: 'Carbon' },
  { symbol: 'N', name: 'Nitrogen' }, { symbol: 'O', name: 'Oxygen' },
  { symbol: 'F', name: 'Fluorine' }, { symbol: 'Ne', name: 'Neon' },
  { symbol: 'Na', name: 'Sodium' }, { symbol: 'Mg', name: 'Magnesium' },
  { symbol: 'Al', name: 'Aluminum' }, { symbol: 'Si', name: 'Silicon' },
  { symbol: 'P', name: 'Phosphorus' }, { symbol: 'S', name: 'Sulfur' },
  { symbol: 'Cl', name: 'Chlorine' }, { symbol: 'Ar', name: 'Argon' },
  { symbol: 'K', name: 'Potassium' }, { symbol: 'Ca', name: 'Calcium' },
  { symbol: 'Sc', name: 'Scandium' }, { symbol: 'Ti', name: 'Titanium' },
  { symbol: 'V', name: 'Vanadium' }, { symbol: 'Cr', name: 'Chromium' },
  { symbol: 'Mn', name: 'Manganese' }, { symbol: 'Fe', name: 'Iron' },
  { symbol: 'Co', name: 'Cobalt' }, { symbol: 'Ni', name: 'Nickel' },
  { symbol: 'Cu', name: 'Copper' }, { symbol: 'Zn', name: 'Zinc' },
  { symbol: 'Ga', name: 'Gallium' }, { symbol: 'Ge', name: 'Germanium' },
  { symbol: 'As', name: 'Arsenic' }, { symbol: 'Se', name: 'Selenium' },
  { symbol: 'Br', name: 'Bromine' }, { symbol: 'Kr', name: 'Krypton' },
  { symbol: 'Rb', name: 'Rubidium' }, { symbol: 'Sr', name: 'Strontium' },
  { symbol: 'Y', name: 'Yttrium' }, { symbol: 'Zr', name: 'Zirconium' },
  { symbol: 'Nb', name: 'Niobium' }, { symbol: 'Mo', name: 'Molybdenum' },
  { symbol: 'Tc', name: 'Technetium' }, { symbol: 'Ru', name: 'Ruthenium' },
  { symbol: 'Rh', name: 'Rhodium' }, { symbol: 'Pd', name: 'Palladium' },
  { symbol: 'Ag', name: 'Silver' }, { symbol: 'Cd', name: 'Cadmium' },
  { symbol: 'In', name: 'Indium' }, { symbol: 'Sn', name: 'Tin' },
  { symbol: 'Sb', name: 'Antimony' }, { symbol: 'Te', name: 'Tellurium' },
  { symbol: 'I', name: 'Iodine' }, { symbol: 'Xe', name: 'Xenon' },
  { symbol: 'Cs', name: 'Cesium' }, { symbol: 'Ba', name: 'Barium' },
  { symbol: 'La', name: 'Lanthanum' }, { symbol: 'Ce', name: 'Cerium' },
  { symbol: 'Pr', name: 'Praseodymium' }, { symbol: 'Nd', name: 'Neodymium' },
  { symbol: 'Pm', name: 'Promethium' }, { symbol: 'Sm', name: 'Samarium' },
  { symbol: 'Eu', name: 'Europium' }, { symbol: 'Gd', name: 'Gadolinium' },
  { symbol: 'Tb', name: 'Terbium' }, { symbol: 'Dy', name: 'Dysprosium' },
  { symbol: 'Ho', name: 'Holmium' }, { symbol: 'Er', name: 'Erbium' },
  { symbol: 'Tm', name: 'Thulium' }, { symbol: 'Yb', name: 'Ytterbium' },
  { symbol: 'Lu', name: 'Lutetium' }, { symbol: 'Hf', name: 'Hafnium' },
  { symbol: 'Ta', name: 'Tantalum' }, { symbol: 'W', name: 'Tungsten' },
  { symbol: 'Re', name: 'Rhenium' }, { symbol: 'Os', name: 'Osmium' },
  { symbol: 'Ir', name: 'Iridium' }, { symbol: 'Pt', name: 'Platinum' },
  { symbol: 'Au', name: 'Gold' }, { symbol: 'Hg', name: 'Mercury' },
  { symbol: 'Tl', name: 'Thallium' }, { symbol: 'Pb', name: 'Lead' },
  { symbol: 'Bi', name: 'Bismuth' }, { symbol: 'Po', name: 'Polonium' },
  { symbol: 'At', name: 'Astatine' }, { symbol: 'Rn', name: 'Radon' },
  { symbol: 'Fr', name: 'Francium' }, { symbol: 'Ra', name: 'Radium' },
  { symbol: 'Ac', name: 'Actinium' }, { symbol: 'Th', name: 'Thorium' },
  { symbol: 'Pa', name: 'Protactinium' }, { symbol: 'U', name: 'Uranium' },
  { symbol: 'Np', name: 'Neptunium' }, { symbol: 'Pu', name: 'Plutonium' },
  { symbol: 'Am', name: 'Americium' }, { symbol: 'Cm', name: 'Curium' },
  { symbol: 'Bk', name: 'Berkelium' }, { symbol: 'Cf', name: 'Californium' },
  { symbol: 'Es', name: 'Einsteinium' }, { symbol: 'Fm', name: 'Fermium' },
  { symbol: 'Md', name: 'Mendelevium' }, { symbol: 'No', name: 'Nobelium' },
  { symbol: 'Lr', name: 'Lawrencium' }, { symbol: 'Rf', name: 'Rutherfordium' },
  { symbol: 'Db', name: 'Dubnium' }, { symbol: 'Sg', name: 'Seaborgium' },
  { symbol: 'Bh', name: 'Bohrium' }, { symbol: 'Hs', name: 'Hassium' },
  { symbol: 'Mt', name: 'Meitnerium' }, { symbol: 'Ds', name: 'Darmstadtium' },
  { symbol: 'Rg', name: 'Roentgenium' }, { symbol: 'Cn', name: 'Copernicium' },
  { symbol: 'Nh', name: 'Nihonium' }, { symbol: 'Fl', name: 'Flerovium' },
  { symbol: 'Mc', name: 'Moscovium' }, { symbol: 'Lv', name: 'Livermorium' },
  { symbol: 'Ts', name: 'Tennessine' }, { symbol: 'Og', name: 'Oganesson' }
]

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

// Socket.IO connection
const connectSocket = () => {
  try {
    socket = io('http://localhost:2518')
    
    socket.on('connect', () => {
      console.log('Connected to Socket.IO server on localhost:2518')
    })
    
    socket.on('disconnect', () => {
      console.log('Disconnected from Socket.IO server')
    })
    
    socket.on('connect_error', (error) => {
      console.error('Socket.IO connection error:', error)
    })

    socket.on('one_click_started', (data) => {
    console.log('Processing started:', data.message);
    });

    // Listen for progress updates
    socket.on('one_click_progress', (data) => {
        console.log('Progress:', data.message);
    });

    // Listen for final results
    socket.on('one_click_result', (data) => {
        console.log('Results received:', data);
    });

    // Listen for errors
    socket.on('one_click_error', (data) => {
        console.error('Error:', data.error);
    });
  } catch (error) {
    console.error('Failed to connect to Socket.IO server:', error)
  }
}

// One Click Analysis function
const performOneClickAnalysis = async () => {
  if (!currentSpectrum.value || !socket) {
    console.error('No spectrum data or socket connection available')
    return
  }
  
  try {
    const message = {
      wavelength: currentSpectrum.value.wavelengths,
      intensity: currentSpectrum.value.intensities,
      n_sigma: nSigma.value,
      subtract_background: subtractBackground.value,
      elements: selectedElements.value
    }
    
    console.log('Sending analysis request:', message)
    
    // Send the message to the "one_click" event
    socket.emit('one_click', message)
    
    // Wait for response
    // socket.once('one_click_result', (response) => {
    //   console.log('Analysis response received:', response)
    // })
    
    // // Set a timeout for the response
    // setTimeout(() => {
    //   console.log('Analysis request timeout - no response received')
    // }, 100000) // 100 second timeout
    
  } catch (error) {
    console.error('Error performing one-click analysis:', error)
  }
}

// Element selection functions
const filterElements = () => {
  if (!elementSearch.value.trim()) {
    filteredElements.value = periodicTable
  } else {
    const searchTerm = elementSearch.value.toLowerCase()
    filteredElements.value = periodicTable.filter(element => 
      element.symbol.toLowerCase().includes(searchTerm) || 
      element.name.toLowerCase().includes(searchTerm)
    )
  }
}

const toggleElement = (symbol: string) => {
  const index = selectedElements.value.indexOf(symbol)
  if (index > -1) {
    selectedElements.value.splice(index, 1)
  } else {
    selectedElements.value.push(symbol)
  }
  // console.log(selectedElements.value)
  // Update the elements string for the analysis
}

const removeElement = (symbol: string) => {
  const index = selectedElements.value.indexOf(symbol)
  if (index > -1) {
    selectedElements.value.splice(index, 1)
  }
  // console.log(selectedElements.value)
}

const handleElementBlur = () => {
  // Delay hiding dropdown to allow for clicks
  setTimeout(() => {
    showElementDropdown.value = false
  }, 200)
}

onMounted(async () => {
  await nextTick()
  createChart()
  connectSocket()
  // Initialize filtered elements
  filteredElements.value = periodicTable
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

.analysis-controls {
  margin-top: 30px;
  padding: 20px;
  background-color: #f8f9fa;
  border-radius: 8px;
  border-left: 4px solid #4CAF50;
}

.analysis-controls h3 {
  margin-top: 0;
  margin-bottom: 15px;
  color: #333;
  font-size: 18px;
}

.control-group {
  margin-bottom: 15px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.control-group label {
  font-size: 16px;
  color: #333;
  min-width: 120px;
}

.number-input, .checkbox-input, .text-input {
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 16px;
  flex-grow: 1;
}

.checkbox-input {
  width: auto;
  min-width: 20px;
  height: 20px;
}

.analysis-button {
  background-color: #007bff;
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

.analysis-button:hover {
  background-color: #0056b3;
}

.analysis-button:active {
  background-color: #004085;
}

.analysis-button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
  color: #888;
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

/* Element selector styles */
.elements-selector {
  position: relative;
  flex-grow: 1;
}

.element-search-input {
  width: 100%;
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 16px;
}

.elements-dropdown {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  max-height: 200px;
  overflow-y: auto;
  background: white;
  border: 1px solid #ccc;
  border-radius: 4px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  z-index: 1000;
}

.element-option {
  padding: 8px 12px;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #eee;
}

.element-option:hover {
  background-color: #f5f5f5;
}

.element-option.selected {
  background-color: #e3f2fd;
}

.element-symbol {
  font-weight: bold;
  color: #333;
}

.element-name {
  color: #666;
  font-size: 14px;
}

.selected-elements {
  margin-top: 8px;
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.selected-element-tag {
  background-color: #4CAF50;
  color: white;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 12px;
  display: flex;
  align-items: center;
  gap: 4px;
}

.remove-element-btn {
  background: none;
  border: none;
  color: white;
  cursor: pointer;
  font-size: 16px;
  padding: 0;
  margin: 0;
  line-height: 1;
}

.remove-element-btn:hover {
  color: #ffebee;
}
</style>
