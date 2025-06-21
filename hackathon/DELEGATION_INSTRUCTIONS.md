# GridPilot-GT: Engineering Delegation Instructions

## 🎯 Project Status: READY FOR PARALLEL DEVELOPMENT

All foundational work is complete! The project scaffold is set up and the integration is tested. You can now start parallel development immediately.

## 📋 Pre-Delegation Checklist ✅

- ✅ Repository initialized and synced
- ✅ Dependencies locked and installed (`requirements.txt`)
- ✅ Complete module scaffolding with working stubs
- ✅ CI pipeline configured (GitHub Actions)
- ✅ Feature branches created for parallel development
- ✅ Integration tested end-to-end
- ✅ All modules pass basic smoke tests

## 🚀 Quick Start for Each Engineer

### For Engineer B (Lane B: Bidding & MPC)
```bash
git clone https://github.com/jadenfix/hackathon.git
cd hackathon
git checkout lane-b-bidding-mpc
pip install -r requirements.txt
python main.py --simulate 1  # Verify system works
```

**Your Mission**: Implement sophisticated bidding strategies and Model Predictive Control
- **Primary Files**: `game_theory/bid_generators.py`, `game_theory/mpc_controller.py`
- **Contract**: See `README_CURSOR.md` → Lane B section
- **Dependencies**: CVXPY for optimization, comprehensive backtesting

### For Engineer C (Lane C: Auction & Dispatch)
```bash
git clone https://github.com/jadenfix/hackathon.git
cd hackathon
git checkout lane-c-auction-dispatch
pip install -r requirements.txt
python main.py --simulate 1  # Verify system works
```

**Your Mission**: Implement VCG auction mechanisms and real-time dispatch
- **Primary Files**: `game_theory/vcg_auction.py`, `dispatch/dispatch_agent.py`
- **Contract**: See `README_CURSOR.md` → Lane C section
- **Focus**: Real-time performance (<100ms), safety protocols

### For Engineer D (Lane D: UI & LLM)
```bash
git clone https://github.com/jadenfix/hackathon.git
cd hackathon
git checkout lane-d-ui-llm
pip install -r requirements.txt
python main.py --simulate 1  # Verify system works
```

**Your Mission**: Build beautiful dashboard and LLM integration
- **Primary Files**: `ui/dashboard.py`, `llm_integration/chat_interface.py`
- **Contract**: See `README_CURSOR.md` → Lane D section
- **Goal**: Beautiful Streamlit UI + local LLM for insights

## 🔄 Development Workflow

### 1. Daily Standup (Recommended)
- **When**: Start of each development session
- **What**: Quick sync on progress, blockers, integration points
- **Duration**: 5-10 minutes

### 2. Development Cycle
1. **Pull latest**: `git pull origin main` (check for any integration updates)
2. **Work on your branch**: Stay in your lane branch
3. **Test frequently**: Use `python main.py --simulate 1` to test integration
4. **Commit often**: Small, focused commits with clear messages
5. **Push regularly**: `git push` to backup your work

### 3. Integration Points
- **Every 2-4 hours**: Test your changes with main integration
- **Before major changes**: Coordinate with other engineers
- **End of session**: Ensure your code integrates cleanly

### 4. Code Review & Merge
1. **When ready**: Open PR from your lane branch to `main`
2. **Requirements**: All tests pass, lint clean, integration works
3. **Review**: Other engineers review for conflicts
4. **Merge**: Merge to main when approved

## 🔗 Integration Interface Points

### Data Flow Between Lanes
```
Lane A (Data) → Lane B (Bidding) → Lane C (Auction) → Lane D (UI)
      ↓              ↓                ↓              ↑
    API Data    Optimized Bids    Allocations    Insights
```

### Critical Dependencies
- **Lane B depends on**: Lane A forecasting output format
- **Lane C depends on**: Lane B bid vector format
- **Lane D depends on**: All lanes for dashboard data
- **All lanes depend on**: `config.toml` structure

### Shared Data Formats (DO NOT CHANGE)
- **Price DataFrame**: columns `['timestamp', 'price', 'volume']`
- **Forecast DataFrame**: columns `['timestamp', 'predicted_price', 'σ_energy', 'σ_hash', 'σ_token']`
- **Allocation Dict**: keys `['inference', 'training', 'cooling']`
- **Inventory Dict**: keys `['power_total', 'power_available', 'battery_soc', 'gpu_utilization']`

## 🧪 Testing Strategy

### Unit Tests (Your Responsibility)
- **Minimum**: 80% coverage for your modules
- **Run**: `pytest tests/test_your_module.py -v`
- **Add**: Tests for all major functions

### Integration Tests (Shared Responsibility)
- **Run**: `python main.py --simulate 1` (full integration)
- **Run**: `pytest tests/test_basic.py -v` (basic integration)
- **Add**: Tests for cross-module interactions

### Performance Tests
- **Lane B**: Bid optimization < 1 second
- **Lane C**: Dispatch response < 100ms
- **Lane D**: Dashboard load < 3 seconds

## 🚨 Critical Success Factors

### 1. Stick to Contracts
- **Interface Signatures**: DO NOT change function signatures in `README_CURSOR.md`
- **Data Formats**: DO NOT change shared data structures
- **Return Types**: Match exactly what other modules expect

### 2. Test Integration Early & Often
- **Every commit**: Test that main.py still works
- **Every feature**: Test your module with real data flow
- **Before PR**: Full integration test passes

### 3. Communication Protocol
- **Breaking Changes**: Notify all engineers immediately
- **API Changes**: Must be agreed upon by all affected parties
- **Blockers**: Escalate within 30 minutes

## 📊 Success Metrics

### Technical KPIs
- [ ] All modules integrate seamlessly
- [ ] System processes real market data
- [ ] Performance meets real-time requirements
- [ ] 80%+ test coverage across all modules
- [ ] CI pipeline always green

### Business KPIs
- [ ] UI provides clear, actionable insights
- [ ] Bidding strategy shows profit optimization
- [ ] Auction mechanism ensures fairness
- [ ] System handles edge cases gracefully

## 🆘 Troubleshooting

### Common Issues & Solutions

**Import Errors**:
```bash
# Solution: Ensure you're in the right directory and have dependencies
cd hackathon
pip install -r requirements.txt
```

**Integration Breaks**:
```bash
# Solution: Test against main branch
git checkout main
git pull
python main.py --simulate 1
git checkout your-lane-branch
# Fix your code to match main interfaces
```

**Merge Conflicts**:
```bash
# Solution: Rebase onto main
git fetch origin
git rebase origin/main
# Resolve conflicts, then continue
```

**Performance Issues**:
- Check if you're following the optimization requirements
- Profile your code with `python -m cProfile main.py --simulate 1`
- Focus on the critical path timing requirements

### Getting Help
1. **Technical Issues**: Check `README_CURSOR.md` contracts
2. **Integration Problems**: Test with minimal changes first
3. **Merge Conflicts**: Coordinate with other engineers
4. **Performance**: Profile and optimize critical sections

## 🎁 Bonus Features (If Time Permits)

### Lane B Enhancements
- Machine learning for bid optimization
- Multi-objective optimization (profit + reliability)
- Advanced risk management models

### Lane C Enhancements
- Real-time market monitoring
- Advanced emergency protocols
- Auction mechanism variations

### Lane D Enhancements
- Mobile-responsive design
- Advanced LLM prompting
- Real-time alerting system

## 🏁 Final Integration

### Hour 5-6: Integration Sprint
- All engineers work together on final integration
- Fix any remaining interface issues
- Performance optimization
- End-to-end testing
- Deployment preparation

### Demo Preparation
- Prepare 5-minute demo of your module
- Show key features and innovations
- Highlight technical achievements
- Practice the full system demo

---

## 💪 You've Got This!

The foundation is solid, the contracts are clear, and the system is tested. Focus on implementing your module's core functionality, test integration frequently, and coordinate with your team. 

**Remember**: Perfect is the enemy of good. Aim for working, tested, integrated code over complex features that break the system.

**Good luck building GridPilot-GT! 🚀⚡🎯** 