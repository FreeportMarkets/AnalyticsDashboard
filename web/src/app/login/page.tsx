import { signIn } from '@/auth'

export default function LoginPage() {
  return (
    <main className="flex min-h-screen items-center justify-center px-6">
      <div className="w-full max-w-sm rounded-lg border border-neutral-800 p-8">
        <h1 className="text-lg font-semibold">Freeport Analytics</h1>
        <p className="mt-1 text-sm text-neutral-400">Internal access only.</p>
        <form
          className="mt-6"
          action={async () => {
            'use server'
            await signIn('google', { redirectTo: '/' })
          }}
        >
          <button
            type="submit"
            className="w-full rounded-md bg-neutral-100 px-4 py-2 text-sm font-medium text-neutral-900 hover:bg-white"
          >
            Sign in with Google
          </button>
        </form>
      </div>
    </main>
  )
}
