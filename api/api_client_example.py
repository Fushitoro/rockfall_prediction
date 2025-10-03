#
# import requests
# import json
# import time
#
# # Example client to test the API
# API_BASE_URL = "http://localhost:5000"
#
# def get_simulated_data():
#     """Get simulated sensor data"""
#     response = requests.get(f"{API_BASE_URL}/simulate-sensor-data")
#     return response.json()
#
# def predict_rockfall(data):
#     """Send data for rockfall prediction"""
#     response = requests.post(f"{API_BASE_URL}/predict-rockfall", json=data)
#     return response.json()
#
# def run_real_time_simulation():
#     """Simulate real-time monitoring"""
#     print("Starting real-time rockfall monitoring simulation...")
#
#     for i in range(10):
#         print(f"\n--- Reading {i+1} ---")
#
#         # Get simulated sensor data
#         sensor_data = get_simulated_data()
#         print(f"Sensor data: {sensor_data['slope_height_m']}m height, "
#               f"{sensor_data['slope_angle_deg']}° angle, "
#               f"{sensor_data['rainfall_mm']}mm rain")
#
#         # Get prediction
#         prediction = predict_rockfall(sensor_data)
#         print(f"Prediction: {prediction['risk_level']} "
#               f"(confidence: {prediction['confidence']:.1%})")
#         print(f"Recommendation: {prediction['recommendation']}")
#
#         if prediction['alert_required']:
#             print("🚨 ALERT TRIGGERED! 🚨")
#
#         time.sleep(2)  # Wait 2 seconds
#
# if __name__ == "__main__":
#     run_real_time_simulation()
