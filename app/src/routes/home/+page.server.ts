// src/routes/posts/+page.server.ts
import { error } from '@sveltejs/kit'
import type { PageServerLoad } from './$types'

export const load: PageServerLoad = async ({ locals: { getSession } }) => {
    const session = await getSession()

    if (!session) {
        // the user is not signed in
        throw error(401, { message: 'Unauthorized' })
    }
}
