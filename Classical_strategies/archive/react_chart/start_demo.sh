#!/bin/bash

echo "Starting React Trading Chart Demo"
echo "================================"
echo ""
echo "This will start the React development server."
echo "The chart will open at http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

# Navigate to react_chart directory
cd "$(dirname "$0")"

# Start the development server
npm run dev