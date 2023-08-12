<script lang="ts">
  import { goto } from "$app/navigation";
  import { onMount } from "svelte";

  let testText: string = "";
  let isRecording: boolean = false;
  let mediaRecorder: MediaRecorder | null = null;
  let audioChunks: BlobPart[] = [];
  let userInput: { type: string; key: string; timestamp: number }[] = [];
  let startTime: number | null = null;

  export let data;


  function logout() {
      data.supabase.auth.signOut().then(() => {
        goto("/");
      }).catch(error => {
          console.error("Error logging out:", error);
      });
  }


  onMount(async () => {
    const user = (await data.supabase.auth.getUser()).data.user;

    console.log(user);
    try {
      const response = await fetch("/api/test");
      if (response.ok) {
        testText = await response.text();
      } else {
        console.error("Failed to fetch test text.");
      }
    } catch (error) {
      console.error("Error fetching test text:", error);
    }

    await startRecording();
  });

  function handleKey(event: KeyboardEvent) {
    console.log(event)
    const eventType = event.type;
    const key = event.key;
    const timestamp = event.timeStamp - (startTime || event.timeStamp); // Calculate the relative timestamp

    userInput.push({ type: eventType, key, timestamp });
  }

  async function startRecording() {
    if (!navigator.mediaDevices) {
      console.error("MediaDevices API not available.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });

        // send it to the server at POST /api/test
        const formData = new FormData();
        formData.append("audio", audioBlob);
        formData.append("userInput", JSON.stringify(userInput));
        const response = await fetch("/api/test", {
          method: "POST",
          body: formData,
        });
      };

      mediaRecorder.start();
      isRecording = true;

      const textArea = document.querySelector("textarea");
      if (textArea) {
        textArea.addEventListener("keydown", handleKey);
        textArea.addEventListener("keyup", handleKey);
      }
      startTime = performance.now();
    } catch (error) {
      console.error("Error starting recording:", error);
    }
  }

  function stopRecording() {
    const textArea = document.querySelector("textarea");
    if (textArea) {
      textArea.removeEventListener("keydown", handleKey);
      textArea.removeEventListener("keyup", handleKey);
    }

    if (mediaRecorder) {
      mediaRecorder.stop();
      isRecording = false;
    }
  }
</script>

<main>
    <button on:click={logout}>Logout</button>
    <p>Type the following text. Press "Stop Recording" when you're finished.</p>
    <p>{testText}</p>
    <textarea />
    <button on:click={stopRecording}>Stop Recording</button>
</main>