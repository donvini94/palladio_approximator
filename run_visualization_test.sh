#!/bin/bash
# Script to test the metrics visualization improvements with focus on outlier handling

# Set up color formatting
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Testing metrics visualization improvements ===${NC}"

# Part 1: Standard test case
echo -e "${BLUE}Running standard test with normal distribution errors...${NC}"
python test_metrics_viz.py

# Check if the performance summary file was created
if [ -f "figures/test_viz/performance_summary.md" ]; then
  echo -e "${GREEN}✅ Performance summary generated successfully${NC}"
  echo -e "${GREEN}Preview of performance summary:${NC}"
  head -n 10 figures/test_viz/performance_summary.md
else
  echo -e "${RED}❌ Failed to generate performance summary${NC}"
  exit 1
fi

# Check if visualizations were created
if [ -f "figures/test_viz/error_distribution_context.png" ]; then
  echo -e "${GREEN}✅ Error distribution visualization generated successfully${NC}"
else
  echo -e "${RED}❌ Failed to generate error distribution visualization${NC}"
  exit 1
fi

# Part 2: Test with extreme outliers
echo -e "\n${BLUE}Running test with extreme outliers...${NC}"
python test_outlier_viz.py

# Check if the outlier test performance summary file was created
if [ -f "figures/test_viz_outliers/performance_summary.md" ]; then
  echo -e "${GREEN}✅ Outlier test performance summary generated successfully${NC}"
else
  echo -e "${RED}❌ Failed to generate outlier test performance summary${NC}"
  exit 1
fi

# Check if outlier visualizations were created
if [ -f "figures/test_viz_outliers/error_distribution_context.png" ]; then
  echo -e "${GREEN}✅ Outlier error distribution visualization generated successfully${NC}"
else
  echo -e "${RED}❌ Failed to generate outlier error distribution visualization${NC}"
  exit 1
fi

echo -e "\n${GREEN}All tests passed successfully!${NC}"
echo -e "You can now run a model training with MLflow enabled to test the complete integration."
echo -e "Example command: python train.py --model torch --embedding bert --use_mlflow"
echo -e "\n${BLUE}Test visualizations are available at:${NC}"
echo -e "- Standard test: figures/test_viz/"
echo -e "- Outlier test: figures/test_viz_outliers/"
echo -e "\nCompare the error_distribution_context.png files in both directories to see the improved outlier handling."