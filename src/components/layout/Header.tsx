import { useQuery } from '@tanstack/react-query'
import { Activity } from 'lucide-react'
import { healthCheck } from '../../services/api'
import { cn } from '../../utils/cn'

export function Header() {
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: healthCheck,
    refetchInterval: 30000,
  })

  const isHealthy = health?.status === 'ok'

  return (
    <header className="bg-white shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <img 
            src="/healthlens-logo.png" 
            alt="HealthLens" 
            className="h-8 w-auto"
            onError={(e) => {
              e.currentTarget.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100" fill="%232563EB"/><text x="50" y="50" font-family="system-ui" font-size="50" fill="white" text-anchor="middle" dominant-baseline="middle">H</text></svg>'
            }}
          />
          <span className="text-xl font-semibold text-gray-900">HealthLens</span>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Activity size={16} className={cn(
              "transition-colors",
              isHealthy ? "text-medical-green" : "text-medical-red"
            )} />
            <span className={cn(
              "text-sm font-medium",
              isHealthy ? "text-medical-green" : "text-medical-red"
            )}>
              {isHealthy ? "System Healthy" : "System Issues"}
            </span>
          </div>
        </div>
      </div>
    </header>
  )
}