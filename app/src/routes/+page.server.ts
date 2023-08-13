// src/routes/+page.server.ts
import { redirect } from '@sveltejs/kit'
import type { PageServerLoad } from './$types'

export const load: PageServerLoad = async ({
    url,
    locals: { getSession, supabase },
}) => {
    const session = await getSession()

    // TODO: this is hacky I think?
    const code = url.searchParams.get('code')

    if (code) {
        await supabase.auth.exchangeCodeForSession(code)
        throw redirect(302, '/home')
    }

    // if the user is already logged in return them to the account page
    if (session) {
        throw redirect(302, '/home')
    }

    return { url: url.origin }
}
