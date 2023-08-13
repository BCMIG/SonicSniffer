<!-- src/routes/+layout.svelte -->
<script lang="ts">
    import '@skeletonlabs/skeleton/themes/theme-skeleton.css'

    // This contains the bulk of Skeletons required styles:
    import '@skeletonlabs/skeleton/styles/skeleton.css'

    // Finally, your application's global stylesheet (sometimes labeled 'app.css')
    import '../app.postcss'

    import { AppBar } from '@skeletonlabs/skeleton'

    import { goto, invalidate } from '$app/navigation'
    import { onMount } from 'svelte'

    import { AppShell } from '@skeletonlabs/skeleton'

    import { LightSwitch } from '@skeletonlabs/skeleton'

    export let data

    let { supabase, session } = data
    $: ({ supabase, session } = data)

    onMount(() => {
        const { data } = supabase.auth.onAuthStateChange((event, _session) => {
            if (_session?.expires_at !== session?.expires_at) {
                invalidate('supabase:auth')
            }
        })

        return () => data.subscription.unsubscribe()
    })

    function logout() {
        supabase.auth
            .signOut()
            .then(() => {
                goto('/')
            })
            .catch((error) => {
                console.error('Error logging out:', error)
            })
    }
</script>

<AppShell>
    <svelte:fragment slot="header">
        <AppShell>
            <svelte:fragment slot="header">
                <AppBar>
                    <svelte:fragment slot="lead">SonicSniffer</svelte:fragment>
                    <svelte:fragment slot="trail">
                        {#if session}
                            <button
                                type="button"
                                class="btn variant-filled"
                                on:click={logout}
                            >
                                <span>Log Out</span>
                            </button>
                        {/if}
                        <LightSwitch />
                    </svelte:fragment>
                </AppBar>
            </svelte:fragment>
            <!-- ... -->
        </AppShell>
    </svelte:fragment>
    <!-- (sidebarLeft) -->
    <!-- (sidebarRight) -->
    <!-- (pageHeader) -->
    <!-- Router Slot -->
    <slot />
    <!-- ---- / ---- -->
    <!-- (pageFooter) -->
    <svelte:fragment slot="footer">Footer</svelte:fragment>
</AppShell>
