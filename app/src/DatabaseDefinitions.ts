export type Json =
    | string
    | number
    | boolean
    | null
    | { [key: string]: Json | undefined }
    | Json[]

export interface Database {
    public: {
        Tables: {
            profiles: {
                Row: {
                    admin: boolean
                    id: string
                }
                Insert: {
                    admin?: boolean
                    id: string
                }
                Update: {
                    admin?: boolean
                    id?: string
                }
                Relationships: [
                    {
                        foreignKeyName: 'profiles_id_fkey'
                        columns: ['id']
                        referencedRelation: 'users'
                        referencedColumns: ['id']
                    },
                ]
            }
            tests: {
                Row: {
                    data: Json | null
                    id: string
                    tid: string
                }
                Insert: {
                    data?: Json | null
                    id: string
                    tid?: string
                }
                Update: {
                    data?: Json | null
                    id?: string
                    tid?: string
                }
                Relationships: [
                    {
                        foreignKeyName: 'tests_id_fkey'
                        columns: ['id']
                        referencedRelation: 'profiles'
                        referencedColumns: ['id']
                    },
                ]
            }
        }
        Views: {
            [_ in never]: never
        }
        Functions: {
            [_ in never]: never
        }
        Enums: {
            [_ in never]: never
        }
        CompositeTypes: {
            [_ in never]: never
        }
    }
}
