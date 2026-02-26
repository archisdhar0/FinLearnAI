import { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, DollarSign, PieChart, Calculator } from 'lucide-react';

// Inflation Calculator
export const InflationCalculator = () => {
  const [amount, setAmount] = useState(10000);
  const [years, setYears] = useState(10);
  const [inflationRate, setInflationRate] = useState(3);

  const futureValue = amount * Math.pow(1 - inflationRate / 100, years);
  const purchasingPowerLost = amount - futureValue;

  return (
    <div className="bg-gradient-to-br from-red-500/10 to-orange-500/10 rounded-xl p-6 border border-red-500/20">
      <div className="flex items-center gap-2 mb-4">
        <TrendingDown className="w-5 h-5 text-red-400" />
        <h4 className="font-semibold text-white">Inflation Impact Calculator</h4>
      </div>
      
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div>
          <label className="text-xs text-slate-400 block mb-1">Starting Amount</label>
          <input
            type="number"
            value={amount}
            onChange={(e) => setAmount(Number(e.target.value))}
            className="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-3 py-2 text-white"
          />
        </div>
        <div>
          <label className="text-xs text-slate-400 block mb-1">Years</label>
          <input
            type="range"
            min="1"
            max="30"
            value={years}
            onChange={(e) => setYears(Number(e.target.value))}
            className="w-full"
          />
          <span className="text-sm text-white">{years} years</span>
        </div>
        <div>
          <label className="text-xs text-slate-400 block mb-1">Inflation Rate</label>
          <input
            type="range"
            min="1"
            max="10"
            step="0.5"
            value={inflationRate}
            onChange={(e) => setInflationRate(Number(e.target.value))}
            className="w-full"
          />
          <span className="text-sm text-white">{inflationRate}%</span>
        </div>
      </div>
      
      <div className="bg-slate-900/50 rounded-lg p-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-slate-400">Today's Value:</span>
          <span className="text-white font-bold">${amount.toLocaleString()}</span>
        </div>
        <div className="flex justify-between items-center mb-2">
          <span className="text-slate-400">Purchasing Power in {years} years:</span>
          <span className="text-red-400 font-bold">${Math.round(futureValue).toLocaleString()}</span>
        </div>
        <div className="flex justify-between items-center pt-2 border-t border-slate-700">
          <span className="text-slate-400">Value Lost to Inflation:</span>
          <span className="text-red-400 font-bold">-${Math.round(purchasingPowerLost).toLocaleString()}</span>
        </div>
      </div>
    </div>
  );
};

// Compound Interest Calculator
export const CompoundCalculator = () => {
  const [principal, setPrincipal] = useState(1000);
  const [monthly, setMonthly] = useState(100);
  const [years, setYears] = useState(20);
  const [rate, setRate] = useState(7);

  const calculateGrowth = () => {
    const r = rate / 100 / 12;
    const n = years * 12;
    
    // Future value of principal
    const fvPrincipal = principal * Math.pow(1 + r, n);
    
    // Future value of monthly contributions
    const fvMonthly = monthly * ((Math.pow(1 + r, n) - 1) / r);
    
    return {
      total: fvPrincipal + fvMonthly,
      contributed: principal + (monthly * n),
      interest: (fvPrincipal + fvMonthly) - (principal + (monthly * n))
    };
  };

  const result = calculateGrowth();

  return (
    <div className="bg-gradient-to-br from-green-500/10 to-emerald-500/10 rounded-xl p-6 border border-green-500/20">
      <div className="flex items-center gap-2 mb-4">
        <TrendingUp className="w-5 h-5 text-green-400" />
        <h4 className="font-semibold text-white">Compound Growth Calculator</h4>
      </div>
      
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label className="text-xs text-slate-400 block mb-1">Starting Amount</label>
          <input
            type="number"
            value={principal}
            onChange={(e) => setPrincipal(Number(e.target.value))}
            className="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-3 py-2 text-white"
          />
        </div>
        <div>
          <label className="text-xs text-slate-400 block mb-1">Monthly Contribution</label>
          <input
            type="number"
            value={monthly}
            onChange={(e) => setMonthly(Number(e.target.value))}
            className="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-3 py-2 text-white"
          />
        </div>
        <div>
          <label className="text-xs text-slate-400 block mb-1">Years: {years}</label>
          <input
            type="range"
            min="1"
            max="40"
            value={years}
            onChange={(e) => setYears(Number(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-xs text-slate-400 block mb-1">Annual Return: {rate}%</label>
          <input
            type="range"
            min="1"
            max="15"
            value={rate}
            onChange={(e) => setRate(Number(e.target.value))}
            className="w-full"
          />
        </div>
      </div>
      
      <div className="bg-slate-900/50 rounded-lg p-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-slate-400">Total Contributed:</span>
          <span className="text-white">${Math.round(result.contributed).toLocaleString()}</span>
        </div>
        <div className="flex justify-between items-center mb-2">
          <span className="text-slate-400">Interest Earned:</span>
          <span className="text-green-400 font-bold">+${Math.round(result.interest).toLocaleString()}</span>
        </div>
        <div className="flex justify-between items-center pt-2 border-t border-slate-700">
          <span className="text-white font-semibold">Final Value:</span>
          <span className="text-green-400 font-bold text-xl">${Math.round(result.total).toLocaleString()}</span>
        </div>
      </div>
      
      {/* Visual bar */}
      <div className="mt-4 h-4 bg-slate-900 rounded-full overflow-hidden flex">
        <div 
          className="bg-blue-500 transition-all"
          style={{ width: `${(result.contributed / result.total) * 100}%` }}
        />
        <div 
          className="bg-green-500 transition-all"
          style={{ width: `${(result.interest / result.total) * 100}%` }}
        />
      </div>
      <div className="flex justify-between text-xs mt-1">
        <span className="text-blue-400">Contributions</span>
        <span className="text-green-400">Interest Growth</span>
      </div>
    </div>
  );
};

// Risk/Return Visualizer
export const RiskReturnChart = () => {
  const investments = [
    { name: 'Savings Account', risk: 5, return: 1, color: '#94a3b8' },
    { name: 'Government Bonds', risk: 15, return: 3, color: '#60a5fa' },
    { name: 'Corporate Bonds', risk: 25, return: 5, color: '#a78bfa' },
    { name: 'Index Funds', risk: 45, return: 8, color: '#4ade80' },
    { name: 'Individual Stocks', risk: 70, return: 12, color: '#fbbf24' },
    { name: 'Crypto', risk: 95, return: 20, color: '#f87171' },
  ];

  return (
    <div className="bg-gradient-to-br from-purple-500/10 to-blue-500/10 rounded-xl p-6 border border-purple-500/20">
      <div className="flex items-center gap-2 mb-4">
        <PieChart className="w-5 h-5 text-purple-400" />
        <h4 className="font-semibold text-white">Risk vs Return</h4>
      </div>
      
      <div className="relative h-64 bg-slate-900/50 rounded-lg p-4">
        {/* Y-axis label */}
        <div className="absolute left-0 top-1/2 -translate-y-1/2 -rotate-90 text-xs text-slate-400">
          Expected Return →
        </div>
        
        {/* X-axis label */}
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 text-xs text-slate-400">
          Risk Level →
        </div>
        
        {/* Plot points */}
        {investments.map((inv) => (
          <div
            key={inv.name}
            className="absolute transform -translate-x-1/2 -translate-y-1/2 group"
            style={{
              left: `${inv.risk}%`,
              bottom: `${(inv.return / 25) * 100}%`,
            }}
          >
            <div
              className="w-4 h-4 rounded-full cursor-pointer transition-transform hover:scale-150"
              style={{ backgroundColor: inv.color }}
            />
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-slate-800 rounded text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity">
              {inv.name}: {inv.return}% return
            </div>
          </div>
        ))}
      </div>
      
      <div className="flex flex-wrap gap-2 mt-4">
        {investments.map((inv) => (
          <div key={inv.name} className="flex items-center gap-1 text-xs">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: inv.color }} />
            <span className="text-slate-400">{inv.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

// Diversification Demo
export const DiversificationDemo = () => {
  const [showDiversified, setShowDiversified] = useState(true);
  
  // Simulated monthly returns
  const singleStock = [0, -5, 8, -12, 15, -8, 3, -20, 25, -3, 10, 5];
  const diversified = [2, -1, 3, -2, 4, -1, 2, -5, 6, 1, 3, 2];
  
  const data = showDiversified ? diversified : singleStock;
  const total = data.reduce((a, b) => a + b, 0);
  
  let cumulative = 100;
  const chartData = data.map(d => {
    cumulative = cumulative * (1 + d / 100);
    return cumulative;
  });

  return (
    <div className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 rounded-xl p-6 border border-blue-500/20">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <DollarSign className="w-5 h-5 text-blue-400" />
          <h4 className="font-semibold text-white">Diversification Effect</h4>
        </div>
        
        <div className="flex gap-2">
          <button
            onClick={() => setShowDiversified(false)}
            className={`px-3 py-1 rounded text-sm ${!showDiversified ? 'bg-red-500 text-white' : 'bg-slate-800 text-slate-400'}`}
          >
            Single Stock
          </button>
          <button
            onClick={() => setShowDiversified(true)}
            className={`px-3 py-1 rounded text-sm ${showDiversified ? 'bg-green-500 text-white' : 'bg-slate-800 text-slate-400'}`}
          >
            Diversified
          </button>
        </div>
      </div>
      
      {/* Simple line chart */}
      <div className="h-32 flex items-end gap-1 bg-slate-900/50 rounded-lg p-4">
        {chartData.map((value, i) => (
          <div
            key={i}
            className={`flex-1 rounded-t transition-all ${showDiversified ? 'bg-green-500' : 'bg-red-500'}`}
            style={{ height: `${Math.max(10, (value / 150) * 100)}%` }}
          />
        ))}
      </div>
      
      <div className="flex justify-between mt-4 text-sm">
        <div>
          <span className="text-slate-400">Starting:</span>
          <span className="text-white ml-2">$100</span>
        </div>
        <div>
          <span className="text-slate-400">Ending:</span>
          <span className={`ml-2 font-bold ${chartData[chartData.length - 1] >= 100 ? 'text-green-400' : 'text-red-400'}`}>
            ${chartData[chartData.length - 1].toFixed(2)}
          </span>
        </div>
        <div>
          <span className="text-slate-400">Volatility:</span>
          <span className={`ml-2 ${showDiversified ? 'text-green-400' : 'text-red-400'}`}>
            {showDiversified ? 'Low' : 'High'}
          </span>
        </div>
      </div>
      
      <p className="text-xs text-slate-400 mt-4">
        {showDiversified 
          ? "A diversified portfolio has smoother returns and lower risk of big losses."
          : "A single stock can have wild swings - both up and down. One bad quarter can wipe out gains."}
      </p>
    </div>
  );
};

export default {
  InflationCalculator,
  CompoundCalculator,
  RiskReturnChart,
  DiversificationDemo
};
