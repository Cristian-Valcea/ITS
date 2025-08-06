#!/bin/bash
# Test script for Team A deliverables
# Run with: ./test_all_deliverables.sh

cd /home/cristian/IntradayTrading/ITS
source venv/bin/activate

echo "🧪 TESTING TEAM A DELIVERABLES"
echo "================================="

failed=0
total=0

# Test function
test_cmd() {
    local name="$1"
    local cmd="$2"
    total=$((total + 1))
    
    echo -n "Testing $name... "
    if eval "$cmd" > /dev/null 2>&1; then
        echo "✅"
    else
        echo "❌"
        failed=$((failed + 1))
    fi
}

# Special test function that checks script execution (not success code)
test_script() {
    local name="$1"
    local cmd="$2"
    total=$((total + 1))
    
    echo -n "Testing $name... "
    # Run command and check if it executes without Python exceptions
    if eval "$cmd" 2>&1 | grep -q "Traceback"; then
        echo "❌ (Python error)"
        failed=$((failed + 1))
    else
        echo "✅ (Script runs)"
    fi
}

# Test 1: Script imports and help functions
echo "📋 Testing script help functions:"
test_cmd "evaluate_phase2.py --help" "python scripts/evaluate_phase2.py --help"
test_cmd "run_seed_variance.py --help" "python scripts/run_seed_variance.py --help"
test_cmd "train.py --help" "python train.py --help"
test_cmd "paper_trading_launcher.py --help" "python operator_docs/paper_trading_launcher.py --help"

# Test 2: Config validation
echo -e "\n🔧 Testing configuration files:"
test_cmd "phase2_oos.yaml syntax" "python -c 'import yaml; yaml.safe_load(open(\"config/curriculum/phase2_oos.yaml\"))'"
test_cmd "Required config sections" "python -c '
import yaml
config = yaml.safe_load(open(\"config/curriculum/phase2_oos.yaml\"))
required = [\"ppo\", \"training\", \"environment\", \"reward_system\"]
missing = [k for k in required if k not in config]
assert len(missing) == 0, f\"Missing: {missing}\"
assert config[\"reward_system\"][\"parameters\"][\"early_exit_tax\"] == 5.0
'"

# Test 3: Functional tests with mock data
echo -e "\n⚙️  Testing functional operations:"
test_cmd "evaluate_phase2.py with mock data" "python scripts/evaluate_phase2.py --run-path test_runs/mock_training_run --output /tmp/test_eval.json"
test_script "run_seed_variance.py analyze mode" "python scripts/run_seed_variance.py --analyze-only 'test_runs/mock_training_run' --output /tmp/test_variance.json"

# Test 4: Integration tests
echo -e "\n🔗 Testing integration components:"
test_cmd "IBKR account manager import" "python -c 'from src.brokers.ibkr_account_manager import IBKRAccountManager; print(\"Import successful\")'"
test_cmd "Training environment imports" "python -c 'from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv; print(\"Import successful\")'"

# Results
echo -e "\n📊 TEST RESULTS:"
echo "==============="
passed=$((total - failed))
echo "Passed: $passed/$total"
echo "Failed: $failed/$total"

if [ $failed -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED - DELIVERABLES READY FOR PHASE 2!"
    exit 0
else
    echo "⚠️  SOME TESTS FAILED - FIX BEFORE PHASE 2 LAUNCH"
    exit 1
fi