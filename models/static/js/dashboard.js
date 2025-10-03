class RockfallDashboard {
  constructor() {
    this.isConnected = false;
    this.zoneMap = ['Zone_A', 'Zone_B', 'Zone_C'];
    this.updateZoneStatus = { Zone_A: 'safe', Zone_B: 'safe', Zone_C: 'safe' };
    this.updateZoneLastNotified = {};
    this.updateNotificationCooldown = 60000; // 1 min per zone
    this.updateInterval = 5000; // 5 seconds
    this.charts = {};
    this.lastUpdate = null;
    this.history = [];
    this.init();
  }

  init() {
    console.log('Initializing Rockfall Dashboard...');
    this.setupCharts();
    this.startDataFetching();
    this.hideLoadingOverlay();
    this.renderPredictionTableHeader();
    this.displayLegend();

    if (window.Notification && Notification.permission !== 'granted') {
      Notification.requestPermission();
    }

    this.renderZones();
  }
displayLegend() {
  const legendContainer = document.createElement('div');
  legendContainer.className = 'legend-container';

  legendContainer.innerHTML = `
    <div class="legend-title">Prediction Status</div>
    <div class="legend-item">
      <span class="badge bg-success legend-badge"></span>Low Risk
    </div>
    <div class="legend-item">
      <span class="badge bg-warning legend-badge"></span>Medium Risk
    </div>
    <div class="legend-item">
      <span class="badge bg-danger legend-badge"></span>High / Critical Risk
    </div>
    <p class="legend-note">
      <strong>Note:</strong><br />
      Risk Level shows the binary decision for immediate alerting.<br />
      Classification shows the multiclass detailed risk category.
    </p>
  `;

  // Insert legend above the prediction table card body
  const predictionsSection = document.querySelector('section > .card .table-responsive');
  if (predictionsSection) {
    predictionsSection.parentElement.insertBefore(legendContainer, predictionsSection);
  } else {
    // fallback at top of container
    const container = document.querySelector('.container-fluid');
    if (container) container.insertBefore(legendContainer, container.firstChild);
  }
}


  renderPredictionTableHeader() {
    const thead = document.querySelector('#predictions-table').closest('table').querySelector('thead tr');
    if (!thead) return;
    thead.innerHTML = `
      <th>Timestamp</th>
      <th class="tooltip" tabindex="0" aria-describedby="tooltip-risklevel">
        Risk Level
        <span class="tooltiptext" role="tooltip" id="tooltip-risklevel">
          Binary model output: overall rockfall risk level for action.
        </span>
      </th>
      <th>Confidence</th>
      <th class="tooltip" tabindex="0" aria-describedby="tooltip-classification">
        Classification
        <span class="tooltiptext" role="tooltip" id="tooltip-classification">
          Multiclass model output: detailed risk category or condition.
        </span>
      </th>
      <th>Key Factors</th>
    `;
  }



  renderZones() {
    const zoneBar = document.getElementById('zone-status-bar');
    if (!zoneBar) return;
    zoneBar.innerHTML = '';
    this.zoneMap.forEach((zone) => {
      const status = this.updateZoneStatus[zone];
      const colorClass =
        status === 'safe'
          ? 'zone-block zone-safe'
          : status === 'medium'
          ? 'zone-block zone-medium'
          : 'zone-block zone-danger';
      let statusText =
        status === 'safe' ? 'SAFE' : status === 'medium' ? 'MONITOR' : 'BLOCKED';
      const div = document.createElement('div');
      div.className = colorClass;
      div.setAttribute('tabindex', '0');
      div.setAttribute('role', 'region');
      div.setAttribute('aria-live', 'polite');
      div.setAttribute('aria-label', `${zone} status: ${statusText}`);
      div.innerHTML = `${zone}: <span>${statusText}</span>`;
      zoneBar.appendChild(div);
    });
  }

  assignZoneAndNotify(riskLevel) {
    const nowIdx = Math.floor(Date.now() / this.updateInterval) % this.zoneMap.length;
    const activeZone = this.zoneMap[nowIdx];

    let zoneStatus = 'safe';
    if (riskLevel === 'medium') zoneStatus = 'medium';
    if (riskLevel === 'high' || riskLevel === 'critical') zoneStatus = 'danger';
    this.zoneMap.forEach((zone) => {
      this.updateZoneStatus[zone] = zone === activeZone ? zoneStatus : 'safe';
    });
    this.renderZones();
    if (zoneStatus === 'danger') this.checkAndNotifyRisk(activeZone, riskLevel);
  }

  checkAndNotifyRisk(zoneKey, riskLevel) {
    const now = Date.now();
    if (
      !this.updateZoneLastNotified[zoneKey] ||
      now - this.updateZoneLastNotified[zoneKey] > this.updateNotificationCooldown
    ) {
      this.sendNotification('Rockfall Danger Zone!', {
        body: `Work in ${zoneKey} BLOCKED. High rockfall risk, stop work.`,
        icon: 'https://user-gen-media-assets.s3.amazonaws.com/seedream_images/507289e3-cbeb-4ca6-939b-3de4554049cb.png',
      });
      this.updateZoneLastNotified[zoneKey] = now;
      console.log(`Work in ${zoneKey} is blocked.`);
    }
  }

  sendNotification(title, options) {
    if (window.Notification && Notification.permission === 'granted') {
      new Notification(title, options);
    } else {
      console.log('Notification:', title, options.body);
    }
  }

  updatePredictionDisplay(binary) {
    if (!binary) return;
    const riskLevel = binary.risk_level.toLowerCase();

    const riskLevelElem = document.getElementById('risk-level');
    const riskCircle = document.getElementById('risk-circle');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');
    const recommendation = document.getElementById('recommendation');

    if (riskLevelElem) riskLevelElem.textContent = binary.risk_level;
    if (riskCircle) {
      riskCircle.className = 'risk-circle ' + riskLevel;
      riskCircle.setAttribute('aria-label', `Current risk level is ${binary.risk_level}`);
    }
    if (confidenceBar && confidenceText) {
      const confPercent = binary.confidence * 100;
      confidenceBar.style.width = confPercent + '%';
      confidenceText.textContent = confPercent.toFixed(1) + '%';
      confidenceBar.className = 'progress-bar ' + this.getRiskColorClass(binary.risk_level);
      confidenceBar.setAttribute('aria-valuenow', confPercent.toFixed(1));
    }
    if (recommendation) {
      recommendation.textContent = binary.recommendation;
      recommendation.setAttribute('aria-label', `Recommendation: ${binary.recommendation}`);
    }

    this.assignZoneAndNotify(riskLevel);

    if (binary.confidence > 0.6)
      this.showAlert(binary.risk_level, binary.recommendation, binary.risk_level);
  }

  getRiskColorClass(riskLevel) {
    switch (riskLevel.toLowerCase()) {
      case 'low':
        return 'bg-success';
      case 'medium':
        return 'bg-warning';
      case 'high':
      case 'critical':
        return 'bg-danger';
      default:
        return 'bg-secondary';
    }
  }

  hideLoadingOverlay() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
      overlay.setAttribute('aria-hidden', 'true');
      overlay.style.opacity = '0';
      setTimeout(() => (overlay.style.display = 'none'), 300);
    }
  }

  setupCharts() {
    const riskCtx = document.getElementById('riskChart').getContext('2d');
    this.charts.risk = new Chart(riskCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Risk Confidence',
            data: [],
            borderColor: '#dc3545',
            backgroundColor: 'rgba(220, 53, 69, 0.15)',
            tension: 0.4,
            fill: true,
            pointRadius: 2,
            pointHoverRadius: 5,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', intersect: false },
        scales: {
          y: {
            beginAtZero: true,
            max: 1,
            ticks: {
              callback: (value) => (value * 100).toFixed(0) + '%',
            },
            grid: {
              drawBorder: false,
              color: '#f0f0f0',
            },
          },
          x: {
            type: 'time',
            time: {
              unit: 'minute',
              displayFormats: { minute: 'HH:mm' },
            },
            grid: {
              drawBorder: false,
              color: '#f0f0f0',
            },
          },
        },
        plugins: {
          legend: { display: true, position: 'top' },
          title: { display: true, text: 'Risk Confidence Over Time', font: { size: 16 } },
          tooltip: {
            enabled: true,
            mode: 'nearest',
            intersect: false,
            callbacks: {
              label: (context) => {
                return context.dataset.label + ': ' + (context.parsed.y * 100).toFixed(2) + '%';
              },
            },
          },
        },
      },
    });

    const sensorCtx = document.getElementById('sensorChart').getContext('2d');
    this.charts.sensor = new Chart(sensorCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Slope Height (m)',
            data: [],
            borderColor: '#007bff',
            backgroundColor: 'rgba(0,123,255,0.2)',
            yAxisID: 'y',
            pointRadius: 2,
            pointHoverRadius: 5,
            borderWidth: 2,
          },
          {
            label: 'Rainfall (mm)',
            data: [],
            borderColor: '#28a745',
            backgroundColor: 'rgba(40,167,69,0.2)',
            yAxisID: 'y1',
            pointRadius: 2,
            pointHoverRadius: 5,
            borderWidth: 2,
          },
          {
            label: 'Vibration Intensity',
            data: [],
            borderColor: '#ffc107',
            backgroundColor: 'rgba(255,193,7,0.2)',
            yAxisID: 'y2',
            pointRadius: 2,
            pointHoverRadius: 5,
            borderWidth: 2,
          },
          {
            label: 'RQD (%)',
            data: [],
            borderColor: '#6f42c1',
            backgroundColor: 'rgba(111,66,193,0.2)',
            yAxisID: 'y3',
            pointRadius: 2,
            pointHoverRadius: 5,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: { type: 'time', time: { unit: 'minute', displayFormats: { minute: 'HH:mm' } } },
          y: {
            type: 'linear',
            display: true,
            position: 'left',
            title: { display: true, text: 'Height (m)' },
            grid: { drawBorder: false },
          },
          y1: { display: false, grid: { drawOnChartArea: false } },
          y2: { display: false, grid: { drawOnChartArea: false } },
          y3: { display: false, grid: { drawOnChartArea: false } },
        },
        plugins: {
          legend: { display: true, position: 'top', labels: { boxWidth: 12, padding: 10 } },
          title: { display: true, text: 'Key Sensor Trends', font: { size: 16 } },
          tooltip: { enabled: true, mode: 'nearest', intersect: false },
        },
      },
    });
  }

  updateSensorDisplay(sensorData) {
    if (!sensorData) return;
    const mapping = {
      'slope-height': sensorData.slope_height_m?.toFixed(1) + ' m',
      'slope-angle': sensorData.slope_angle_deg?.toFixed(1) + ' °',
      rainfall: sensorData.rainfall_mm?.toFixed(1) + ' mm',
      vibration: sensorData.vibration_intensity?.toFixed(2),
      cohesion: sensorData.cohesion_kpa?.toFixed(1) + ' kPa',
      temperature: sensorData.temperature_range_c?.toFixed(1) + ' °C',
      groundwater: sensorData.groundwater_depth_m?.toFixed(1) + ' m',
      rqd: sensorData.rqd_percent?.toFixed(1) + ' %',
    };

    Object.entries(mapping).forEach(([id, val]) => {
      const el = document.getElementById(id);
      if (el && val !== undefined) {
        el.textContent = val;
        el.classList.add('updated');
        setTimeout(() => el.classList.remove('updated'), 600);
      }
    });

    this.lastUpdate = new Date();
    this.updateLastUpdateTime();
  }

  updateCharts(data) {
    if (!data || data.length === 0 || !this.charts.risk || !this.charts.sensor) return;

    const riskLabels = data.map((d) => new Date(d.timestamp));
    const riskData = data.map((d) => d.prediction.binary_result?.confidence || 0);

    this.charts.risk.data.labels = riskLabels;
    this.charts.risk.data.datasets[0].data = riskData;
    this.charts.risk.update('none');

    const sensorLabels = data.map((d) => new Date(d.timestamp));
    this.charts.sensor.data.labels = sensorLabels;

    this.charts.sensor.data.datasets[0].data = data.map((d) => d.sensor_data?.slope_height_m || 0);
    this.charts.sensor.data.datasets[1].data = data.map((d) => d.sensor_data?.rainfall_mm || 0);
    this.charts.sensor.data.datasets[2].data = data.map((d) => d.sensor_data?.vibration_intensity || 0);
    this.charts.sensor.data.datasets[3].data = data.map((d) => d.sensor_data?.rqd_percent || 0);

    this.charts.sensor.update('none');
  }

  async fetchLatestAndUpdate() {
    try {
      const response = await fetch('/api/latest');
      const data = await response.json();
      if (data.sensor_data) this.updateSensorDisplay(data.sensor_data);
      if (data.prediction && data.prediction.binary_result) this.updatePredictionDisplay(data.prediction.binary_result);
      this.setConnectionStatus(true);
      if (data.timestamp) {
        this.history.push(data);
        if (this.history.length > 50) this.history.shift();
        this.updateCharts(this.history);
        this.updatePredictionsTable(this.history);
      }
    } catch (error) {
      console.error('Error fetching latest data:', error);
      this.setConnectionStatus(false);
    }
  }

  async fetchHistory() {
    try {
      const response = await fetch('/api/history');
      const data = await response.json();
      this.history = data;
      this.updateCharts(data);
      this.updatePredictionsTable(data);
    } catch (error) {
      console.error('Error fetching history:', error);
    }
  }

  updatePredictionsTable(predictionEntries) {
    const tbody = document.getElementById('predictions-table');
    if (!tbody || !predictionEntries || predictionEntries.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="text-center py-3">No predictions available</td></tr>';
      return;
    }
    tbody.innerHTML = '';

    const labelMap = {
      '0': 'Low',
      '1': 'Medium',
      '2': 'High',
      '3': 'Critical',
    };

    const recentEntries = predictionEntries.slice(-10).reverse();

    recentEntries.forEach((entry) => {
      const timestampStr = entry.timestamp || '';
      const timestamp = timestampStr ? new Date(timestampStr).toLocaleString() : 'N/A';

      const binary = entry.prediction?.binary_result || {};
      const multiclass = entry.prediction?.multiclass_result || {};

      const riskLevel = binary.risk_level || 'UNKNOWN';
      const riskBadge = `<span class="badge ${this.getRiskColorClass(riskLevel)}">${riskLevel}</span>`;
      const confidence =
        binary.confidence !== undefined ? `${(binary.confidence * 100).toFixed(1)}%` : 'N/A';

      const rawLabel = multiclass.prediction_label || 'N/A';
      const classification = labelMap[rawLabel] || rawLabel;

      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${timestamp}</td>
        <td>${riskBadge}</td>
        <td>${confidence}</td>
        <td>${classification}</td>
        <td><small class="text-muted">Slope conditions analyzed</small></td>
      `;
      tbody.appendChild(row);
    });
  }

  setConnectionStatus(connected) {
    this.isConnected = connected;
    const statusElem = document.getElementById('connection-status');
    if (statusElem) {
      statusElem.textContent = connected ? 'Connected' : 'Disconnected';
      statusElem.className = `badge me-3 ${connected ? 'bg-success' : 'bg-danger'}`;
    }
  }

  updateLastUpdateTime() {
    const lastUpdateElem = document.getElementById('last-update');
    if (lastUpdateElem && this.lastUpdate) {
      lastUpdateElem.textContent = `Last update: ${this.lastUpdate.toLocaleTimeString()}`;
    }
  }

  startDataFetching() {
    this.fetchLatestAndUpdate();
    this.fetchHistory();
    setInterval(() => {
      this.fetchLatestAndUpdate();
      this.fetchHistory();
    }, this.updateInterval);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.dashboard = new RockfallDashboard();
});
