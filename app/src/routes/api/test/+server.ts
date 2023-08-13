import { randomUUID } from 'crypto'

export async function GET() {
    const response = await fetch(
        'https://random-word-api.herokuapp.com/word?number=50'
    )

    const test = (await response.json()).join(' ')

    console.log(test)

    return new Response(test)
}

// upload a completed test
// completed tests contain a list of all keyup and keydown events with times and an audio file in wav format
export async function POST({ request, locals: { supabase, getSession } }) {
    const session = await getSession()

    if (!session) {
        return new Response()
    }

    const id = session.user.id

    const data = await request.formData()

    const events = data.get('userInput')
    const file = data.get('audio') as File

    // Assuming each user has a unique path, else you might want to give the file a unique name.
    const filePath = `audio/${id}/${randomUUID()}.wav`

    const { error } = await supabase.storage
        .from('files')
        .upload(filePath, file)

    if (error) {
        console.error('Error uploading audio:', error)
        return new Response()
    }

    const { error: testError } = await supabase.from('tests').insert({
        id: id,
        data: events,
    })

    if (testError) {
        console.error('Error uploading test:', testError)
    }

    return new Response()
}
