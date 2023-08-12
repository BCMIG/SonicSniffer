import { readFileSync } from "fs";

import { randomUUID } from "crypto";

export function GET() {
    const words = readFileSync('src/routes/api/test/words.txt', 'utf8').split('\n')
    const test = words.sort(() => Math.random() - Math.random()).slice(0, 50).join(' ')
    
    return new Response(test);
}

// upload a completed test
// completed tests contain a list of all keyup and keydown events with times and an audio file in wav format
export async function POST({request, cookies, locals: { supabase, getSession }}) {
    const session = await getSession();

    if (!session) {
        return new Response()
    }

    const id = session.user.id;

    const data = await request.formData();

    const events = data.get("userInput");
    const file = data.get("audio") as File;

    // Assuming each user has a unique path, else you might want to give the file a unique name.
    const filePath = `${id}/${randomUUID()}.wav`;  

    const { error } = await supabase.storage.from("files").upload(filePath, file);

    if (error) {
        console.error("Error uploading audio:", error);
        return new Response()
    }

    return new Response()
}