"use client";

import { useAuth } from "@/lib/auth-context";
import { Navbar } from "@/components/navbar";
import { Sidebar } from "@/components/sidebar";
import ChatWindow from "@/app/chat/chat-window";
import { useChat } from "@/lib/ChatContext";

export default function ChatPage() {
  const { user, isLoading } = useAuth();
  // A1: all chat state lives in ChatContext — page.tsx is now a thin layout shell
  const { chats, currentChatId, startNewChat, loadChat, deleteChat } = useChat();

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 rounded-full border-4 border-primary border-t-transparent animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <div className="flex flex-col h-screen">
      <Navbar />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          chats={chats}
          activeChat={currentChatId}
          onNewChat={startNewChat}
          onSelectChat={loadChat}
          onDeleteChat={deleteChat}
        />
        <div className="flex-1 flex flex-col">
          <ChatWindow onNewChat={startNewChat} />
        </div>
      </div>
    </div>
  );
}
