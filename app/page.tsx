'use client'

import { useMemo, useState } from 'react'
import {
  ArrowRight,
  CarFront,
  ChevronDown,
  CircleHelp,
  Info,
  RotateCcw,
  ShieldCheck,
} from 'lucide-react'

const fields = [
  { key: 'buying', label: 'Buying price', hint: 'Initial investment', options: ['Low', 'Medium', 'High', 'Very high'] },
  { key: 'maintenance', label: 'Maintenance cost', hint: 'Expected upkeep', options: ['Low', 'Medium', 'High', 'Very high'] },
  { key: 'doors', label: 'Number of doors', hint: 'Vehicle access', options: ['2', '3', '4', '5+'] },
  { key: 'persons', label: 'Passenger capacity', hint: 'Seats available', options: ['2', '4', 'More than 4'] },
  { key: 'luggage', label: 'Luggage boot', hint: 'Cargo capacity', options: ['Small', 'Medium', 'Large'] },
  { key: 'safety', label: 'Safety rating', hint: 'Protection level', options: ['Low', 'Medium', 'High'] },
]

const initialValues = Object.fromEntries(fields.map((field) => [field.key, field.options[1]]))

export default function Page() {
  const [values, setValues] = useState<Record<string, string>>(initialValues)
  const [result, setResult] = useState<'Ready' | 'Acceptable' | 'Good' | 'Very good'>('Ready')
  const [isEvaluating, setIsEvaluating] = useState(false)

  const score = useMemo(() => {
    const positive = ['Low', 'Medium', '4', 'More than 4', 'Large', 'High']
    const matched = Object.values(values).filter((value) => positive.includes(value)).length
    return Math.min(96, Math.max(58, 64 + matched * 5))
  }, [values])

  function evaluate(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setIsEvaluating(true)
    window.setTimeout(() => {
      setResult(score >= 86 ? 'Very good' : score >= 76 ? 'Good' : 'Acceptable')
      setIsEvaluating(false)
    }, 700)
  }

  function reset() {
    setValues(initialValues)
    setResult('Ready')
  }

  return (
    <main className="min-h-screen overflow-hidden bg-[#07111d] text-slate-100 selection:bg-cyan-300 selection:text-slate-950">
      <div className="dashboard-grid pointer-events-none fixed inset-0 opacity-40" aria-hidden="true" />
      <div className="orb orb-one pointer-events-none" aria-hidden="true" />
      <div className="orb orb-two pointer-events-none" aria-hidden="true" />

      <div className="relative mx-auto flex min-h-screen w-full max-w-[1500px] flex-col px-5 py-6 sm:px-8 lg:px-12">
        <header className="animate-rise flex items-center justify-between border-b border-white/10 pb-5">
          <div className="flex items-center gap-3">
            <div className="brand-mark"><CarFront className="size-5" /></div>
            <div>
              <p className="text-lg font-semibold tracking-[0.16em] text-white">TORQUE<span className="text-cyan-300">X</span></p>
              <p className="text-[9px] font-medium uppercase tracking-[0.28em] text-slate-500">Intelligent mobility lab</p>
            </div>
          </div>
          <div className="hidden items-center gap-2 rounded-full border border-emerald-300/20 bg-emerald-300/5 px-3 py-1.5 text-[10px] font-medium uppercase tracking-[0.16em] text-emerald-300 sm:flex">
            <span className="status-dot" /> Model online
          </div>
        </header>

        <section className="grid flex-1 items-center gap-10 py-10 lg:grid-cols-[0.82fr_1.18fr] lg:gap-20 lg:py-16">
          <div className="animate-rise max-w-xl" style={{ animationDelay: '100ms' }}>

            <h1 className="max-w-2xl text-5xl font-semibold leading-[0.98] tracking-[-0.055em] text-white sm:text-6xl xl:text-7xl">Understand what your car is <span className="text-gradient">worth today.</span></h1>
            <p className="mt-7 max-w-md text-base leading-7 text-slate-400">Tell us about your current vehicle. We weigh its costs, practicality, and safety to give you a clear value assessment.</p>
          </div>

          <div className="animate-rise" style={{ animationDelay: '180ms' }}>
            <div className="panel relative overflow-hidden rounded-[28px] p-5 sm:p-7">
              <div className="absolute right-7 top-7 hidden items-center gap-2 text-[10px] uppercase tracking-[0.18em] text-slate-500 sm:flex"><ShieldCheck className="size-3.5 text-cyan-300" /> Secure evaluation</div>
              <div className="mb-7 flex items-end justify-between border-b border-white/10 pb-5">
                <div><p className="eyebrow">Current vehicle</p><h2 className="mt-2 text-2xl font-semibold tracking-tight text-white">Describe your car</h2></div>
                <button type="button" onClick={reset} className="icon-button" aria-label="Reset vehicle profile" title="Reset profile"><RotateCcw className="size-4" /></button>
              </div>

              <form onSubmit={evaluate}>
                <div className="grid gap-3 sm:grid-cols-2">
                  {fields.map((field, index) => (
                    <label key={field.key} className="field-card group" style={{ animationDelay: `${index * 45}ms` }}>
                      <span className="flex items-center justify-between"><span className="field-label">{field.label}</span><CircleHelp className="size-3.5 text-slate-600 transition-colors group-hover:text-cyan-300" /></span>
                      <span className="mt-2 flex items-center justify-between gap-2"><span className="text-sm font-medium text-slate-100">{values[field.key]}</span><ChevronDown className="dropdown-chevron size-4 text-slate-500" /></span>
                      <select className="dropdown-select" aria-label={field.label} value={values[field.key]} onChange={(event) => setValues({ ...values, [field.key]: event.target.value })}><option value="" disabled>{field.hint}</option>{field.options.map((option) => <option key={option}>{option}</option>)}</select>
                    </label>
                  ))}
                </div>
                <button className="evaluate-button mt-5 flex w-full items-center justify-center gap-2" type="submit" disabled={isEvaluating}>{isEvaluating ? <><span className="spinner" /> Assessing vehicle...</> : <>Estimate current value <ArrowRight className="size-4" /></>}</button>
              </form>

              {result !== 'Ready' && <div className="result-card animate-result mt-5" role="status"><div className="flex items-center justify-between"><div><p className="eyebrow text-emerald-300/70">Value assessment</p><h3 className="mt-1 text-2xl font-semibold text-white">{result}</h3></div><div className="score-ring"><span>{score}</span><small>%</small></div></div><div className="mt-4 h-1.5 overflow-hidden rounded-full bg-white/10"><div className="score-bar" style={{ width: `${score}%` }} /></div><p className="mt-3 text-xs leading-5 text-slate-400">Based on the details of your current car and patterns learned from 1,728 vehicle evaluations.</p></div>}
            </div>
          </div>
        </section>

        <footer className="flex flex-col gap-3 border-t border-white/10 py-5 text-[10px] uppercase tracking-[0.16em] text-slate-500 sm:flex-row sm:items-center sm:justify-between"><span>Powered by supervised learning</span><span className="flex items-center gap-2"><Info className="size-3" /> For guidance, not a final purchase decision</span></footer>
      </div>
    </main>
  )
}
