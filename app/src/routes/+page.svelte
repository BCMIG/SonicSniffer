<!-- src/routes/+page.svelte -->
<script lang="ts">
    import { Auth } from '@supabase/auth-ui-svelte'
    import { ThemeSupa } from '@supabase/auth-ui-shared'
    import { onMount } from 'svelte'
    let ready = false

    onMount(() => (ready = true))

    export let data
</script>

<svelte:head>
    <title>Authentication</title>
</svelte:head>

{#if ready}
    <label class="label">
        <span>Input</span>
        <input class="input" type="text" placeholder="Input" />
        <button type="button" class="btn btn-lg variant-filled">lg</button>
    </label>
{/if}
<!-- 
{#if ready}
  <div class="row flex-center flex">
    <div class="col-6 form-widget">
      <Auth
        supabaseClient={data.supabase}
        view="magic_link"
        redirectTo={`/auth/callback`}
        providers={["github"]}
        showLinks={false}
        appearance={{ theme: ThemeSupa }}
        additionalData={{ hacky: "this is hacky fix" }}
      />
    </div>
  </div>
{/if} -->
