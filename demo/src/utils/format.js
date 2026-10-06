// Dates arrive as ISO "YYYY-MM-DD" strings. They are parsed by hand so the
// browser timezone can never shift a month (new Date('2020-01-01') is UTC midnight,
// which renders as 31 Dec 2019 west of Greenwich).
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

export function monthIndex(iso) {
  const [year, month] = iso.split('-').map(Number)
  return year * 12 + (month - 1)
}

export function formatMonthIndex(index) {
  return `${MONTHS[((index % 12) + 12) % 12]} ${Math.floor(index / 12)}`
}

export function formatMonth(iso) {
  return iso ? formatMonthIndex(monthIndex(iso)) : '—'
}

export function shiftMonths(iso, delta) {
  return formatMonthIndex(monthIndex(iso) + delta)
}

// Rents are ~2–10 SGD/sqft/month (2 decimals); sale prices are ~500–4000 SGD/sqft (whole dollars).
export function formatValue(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  const n = Number(value)
  return n.toFixed(digits != null ? digits : (Math.abs(n) >= 100 ? 0 : 2))
}

export function formatPct(value) {
  if (value == null) return '—'
  return `${value >= 0 ? '+' : ''}${Number(value).toFixed(1)}%`
}

export function apiError(error) {
  const data = error && error.response && error.response.data
  const detail = data && (data.detail || data.error)
  if (detail) return typeof detail === 'string' ? detail : JSON.stringify(detail)
  if (error && error.response) return `Request failed (${error.response.status})`
  return 'Unable to reach the PGIM API. Is the backend running on port 8000?'
}

export function isCancel(error) {
  return error && (error.name === 'CanceledError' || error.code === 'ERR_CANCELED')
}
