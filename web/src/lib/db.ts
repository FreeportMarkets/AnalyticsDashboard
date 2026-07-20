import { neon } from '@neondatabase/serverless'

const url = process.env.DATABASE_URL
if (!url) throw new Error('DATABASE_URL is not set')

// Explicit type params, not a bare `neon(url)` call. `neon`'s type params have
// defaults, so `ReturnType<typeof neon>` -- the `SqlTag` alias every
// src/lib/sync/* module types its `sql` parameter as -- resolves via the
// params' *constraints* (`boolean`), not their defaults (`false`). A bare
// call infers the narrower `NeonQueryFunction<false, false>`, which is NOT
// assignable to `NeonQueryFunction<boolean, boolean>` (the `transaction`
// method's contravariant parameter makes them structurally incompatible), so
// this export would fail to typecheck against any sync module that consumes
// it. See test/integration.test.ts's identical comment.
export const sql = neon<boolean, boolean>(url)
