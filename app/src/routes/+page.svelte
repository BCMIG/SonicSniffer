<!-- src/routes/+page.svelte -->
<script lang="ts">
	import { Auth } from '@supabase/auth-ui-svelte'
	import { ThemeSupa } from '@supabase/auth-ui-shared'

	export let data
</script>

<svelte:head>
	<title>User Management</title>
</svelte:head>

<div class="row flex-center flex">
	<div class="col-6 form-widget">
		<Auth
			supabaseClient={data.supabase}
			view="magic_link"
			redirectTo={`/auth/callback`}
			providers={['github']}
			showLinks={false}
			appearance={{theme: ThemeSupa}}
			additionalData={{ hacky: "this is hacky fix" }}
		/>
	</div>
</div>