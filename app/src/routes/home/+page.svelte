<script lang="ts">
    import { goto } from '$app/navigation'
    import { onMount } from 'svelte'

    let testText: string = ''
    let isRecording: boolean = false
    let mediaRecorder: MediaRecorder | null = null
    let audioChunks: BlobPart[] = []
    let userInput: { type: string; key: string; timestamp: number }[] = []
    let startTime: number | null = null

    export let data

    function logout() {
        data.supabase.auth
            .signOut()
            .then(() => {
                goto('/')
            })
            .catch((error) => {
                console.error('Error logging out:', error)
            })
    }

    onMount(async () => {
        const user = (await data.supabase.auth.getUser()).data.user
        console.log(user)

        try {
            const response = await fetch('/api/test')
            if (response.ok) {
                testText = await response.text()
            } else {
                console.error('Failed to fetch test text.')
            }
        } catch (error) {
            console.error('Error fetching test text:', error)
        }

        await startRecording()
    })

    function handleKey(event: KeyboardEvent) {
        console.log(event)
        const eventType = event.type
        const key = event.key

        if (startTime === null) {
            startTime = event.timeStamp
        }

        const timestamp = event.timeStamp - startTime
        userInput.push({ type: eventType, key, timestamp })
    }

    async function startRecording(repeat: boolean = false) {
        if (!navigator.mediaDevices) {
            console.error('MediaDevices API not available.')
            return
        }

        if (repeat) {
            userInput = []
            startTime = null

            try {
                const response = await fetch('/api/test')
                if (response.ok) {
                    testText = await response.text()
                } else {
                    console.error('Failed to fetch test text.')
                }
            } catch (error) {
                console.error('Error fetching test text:', error)
            }

            const textArea = document.querySelector('textarea')
            if (textArea) {
                textArea.value = ''
            }
        }

        try {
            // clear the textarea
            console.log('Started recording.')

            const stream = await navigator.mediaDevices.getUserMedia({
                audio: true,
            })
            mediaRecorder = new MediaRecorder(stream)
            audioChunks = []

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data)
                }
            }

            mediaRecorder.onstop = async () => {
                console.log('Stopped recording.')
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' })
                const formData = new FormData()
                formData.append('audio', audioBlob)
                formData.append('userInput', JSON.stringify(userInput))
                const response = await fetch('/api/test', {
                    method: 'POST',
                    body: formData,
                })
            }

            mediaRecorder.start()
            isRecording = true

            const textArea = document.querySelector('textarea')
            if (textArea) {
                textArea.addEventListener('keydown', handleKey)
                textArea.addEventListener('keyup', handleKey)
            }
        } catch (error) {
            console.error('Error starting recording:', error)
        }
    }

    function stopRecording() {
        console.log('Stopping recording.')
        const textArea = document.querySelector('textarea')
        if (textArea) {
            textArea.removeEventListener('keydown', handleKey)
            textArea.removeEventListener('keyup', handleKey)
        }

        if (mediaRecorder) {
            mediaRecorder.stop()
            isRecording = false
        }
    }
</script>

<main>
    <button on:click={logout}>Logout</button>
    <p>Type the following text. Press "Stop Recording" when you're finished.</p>
    <p>{testText}</p>
    <textarea />
    {#if isRecording}
        <button class="btn" on:click={stopRecording}>Stop Recording</button>
    {:else}
        <button class="btn" on:click={() => startRecording(true)}
            >Start Recording</button
        >
    {/if}
    <span class="recording-indicator">
        {#if isRecording}
            <div class="recording-dot" />
            Recording...{/if}
    </span>

    <br />
    <br />

    Jason Adhinarta Will Morrison Jason Adhinarta Will Morrison Jason Adhinarta Will Morrison Jason Adhinarta Will Morrison 
</main>

<style>
    main {
        max-width: 800px;
        margin: 50px auto;
        padding: 20px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    p {
        margin-bottom: 20px;
        line-height: 1.5;
    }

    button {
        background-color: #e0e0e0;
        color: #333;
        border: none;
        padding: 10px 20px;
        margin-right: 10px;
        cursor: pointer;
        transition: background-color 0.2s;
    }

    button:hover {
        background-color: #d0d0d0;
    }

    textarea {
        width: 100%;
        padding: 10px;
        border: 1px solid #ccc;
        resize: vertical;
        min-height: 150px;
    }

    .recording-indicator {
        display: inline-flex;
        align-items: center;
        color: red;
        margin-left: 10px;
        font-weight: bold;
    }

    .recording-dot {
        width: 10px;
        height: 10px;
        background-color: red;
        border-radius: 50%;
        margin-right: 5px;
    }
</style>
