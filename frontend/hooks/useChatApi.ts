/**
 * useChatApi — thin hook wrapper over ChatContext.
 *
 * Keeps component imports stable: components import from this hook and are
 * shielded from direct ChatContext shape changes.
 */
import { useChat } from "@/lib/ChatContext";

export function useChatApi() {
  const {
    messages,
    isLoading,
    streamingMessageId,
    sendUserMessage,
    addMessage,
    sendHiddenMessage,
  } = useChat();

  return {
    messages,
    isLoading,
    streamingMessageId,
    /** Send a user message and stream the response. */
    sendMessage: sendUserMessage,
    addMessage,
    sendHiddenMessage,
  };
}
