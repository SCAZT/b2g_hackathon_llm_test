-- Seed Test Data for Chatbot + RAG Testing
-- Purpose: Create sample user and data for testing

-- ============================================
-- Test User
-- ============================================
INSERT INTO "user" (
    first_name, last_name, email, dob, agent_type, gender,
    education_field, education_level, genai_usage_frequency,
    disability_knowledge, genai_course_exp, token
) VALUES (
    'Test', 'User', 'test@example.com', '1990-01-01', 1, 'prefer_not_to_say',
    'Computer Science', 'Masters', 'Weekly',
    'Some knowledge', 'Yes', 'test-token-12345'
) ON CONFLICT (email) DO NOTHING;

-- Get the user_id for the test user
DO $$
DECLARE
    test_user_id INTEGER;
BEGIN
    SELECT user_id INTO test_user_id FROM "user" WHERE email = 'test@example.com';

    -- ============================================
    -- Sample Conversations (for testing memory creation)
    -- ============================================
    INSERT INTO conversation (user_id, message_type, content, role, mode, agent_type, sequence_number) VALUES
    (test_user_id, 'user_input', '我想为独臂用户设计一个磁性拉链工具', 'user', 'chat', 1, 1),
    (test_user_id, 'agent_response', '这是一个很有意义的设计方向！磁性拉链可以大大简化独臂用户穿衣的过程。让我们一起探讨一下这个创意的细节...', 'assistant', 'chat', 1, 2),
    (test_user_id, 'user_input', '你觉得这个想法的可行性如何？', 'user', 'chat', 1, 3),
    (test_user_id, 'agent_response', '从可行性角度来看，这个想法非常实际。市场上已经有一些磁性拉链产品，但大多是出厂集成的...', 'assistant', 'chat', 1, 4);

END $$;

-- ============================================
-- Verify Seed Data
-- ============================================
SELECT 'Test User Created:' as status, user_id, email, first_name, last_name
FROM "user" WHERE email = 'test@example.com';

SELECT 'Sample Conversations:' as status, COUNT(*) as count
FROM conversation WHERE user_id = (SELECT user_id FROM "user" WHERE email = 'test@example.com');
