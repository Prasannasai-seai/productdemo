import React, { useRef, useEffect } from 'react';
import { ChatMessage as ChatMessageType } from '../../types';
import ChatMessage from './ChatMessage';
import { ChatBubbleLeftRightIcon, ArrowDownIcon } from '@heroicons/react/24/outline';
import { messageListStyles, messageBubbleStyles } from './chatStyles';

// Import RagSource type
import { RagSource } from '../../services/ragChatService';

// Allow for extended message types that include system messages
interface MessageListProps {
  messages: {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    isStreaming?: boolean;
    fileAttachment?: {
      name: string;
      type: string;
      size: number;
      url?: string;
      documentId?: string;
      status?: string;
      processingError?: string;
    };
    isProcessingFile?: boolean;
    isProcessingOnly?: boolean; // Flag to indicate this is document processing, not message streaming
    isLoadingOnly?: boolean; // Flag to indicate this is just a loading indicator with no text
    sources?: RagSource[]; // Add sources for RAG responses
    useRag?: boolean; // Flag to indicate if RAG was used
  }[];
  isLoading: boolean;
  hasMoreMessages: boolean;
  loadMoreMessages: () => void;
  loadingMessages: boolean;
  isEmpty?: boolean;
  onDownloadClick?: () => void;
}

const MessageList: React.FC<MessageListProps> = ({
  messages,
  isLoading,
  hasMoreMessages,
  loadMoreMessages,
  loadingMessages,
  isEmpty = false,
  onDownloadClick
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of messages when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Setup infinite scrolling for messages
  useEffect(() => {
    const messagesContainer = messagesContainerRef.current;
    if (!messagesContainer) return;

    const handleScroll = () => {
      if (messagesContainer.scrollTop === 0 && hasMoreMessages && !loadingMessages) {
        loadMoreMessages();
      }
    };

    messagesContainer.addEventListener('scroll', handleScroll);
    return () => messagesContainer.removeEventListener('scroll', handleScroll);
  }, [messagesContainerRef, hasMoreMessages, loadingMessages, loadMoreMessages]);

  // Group messages by role
  const groupedMessages = React.useMemo(() => {
    const groups: { role: string; messages: ChatMessageType[] }[] = [];

    messages.forEach(message => {
      const lastGroup = groups[groups.length - 1];
      if (lastGroup && lastGroup.role === message.role) {
        lastGroup.messages.push(message);
      } else {
        groups.push({ role: message.role, messages: [message] });
      }
    });

    return groups;
  }, [messages]);

  // Update the handleLinkClick function to detect the new download format
  const handleLinkClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!onDownloadClick) return;
    
    // Check if the click was on a link or button
    const target = e.target as HTMLElement;
    
    // Check for both anchor tags and bold text with our custom download format
    const anchor = target.tagName === 'A' ? target : target.closest('a');
    
    // First check if it's a normal anchor tag with download: prefix
    if (anchor) {
      const href = anchor.getAttribute('href');
      if (href && href.startsWith('download:')) {
        e.preventDefault();
        onDownloadClick();
        return;
      }
    }
    
    // Also check for our special format in text content (for bold text click)
    const textContent = target.textContent || '';
    if (textContent.includes('[DOWNLOAD_ALL_RESULTS:') || 
        target.innerHTML.includes('[DOWNLOAD_ALL_RESULTS:')) {
      e.preventDefault();
      onDownloadClick();
      return;
    }
    
    // Check if any parent has the download text (for when clicking on an icon within the text)
    let current = target;
    for (let i = 0; i < 5; i++) { // Check up to 5 levels up
      if (!current.parentElement) break;
      current = current.parentElement;
      
      const parentText = current.textContent || '';
      if (parentText.includes('[DOWNLOAD_ALL_RESULTS:')) {
        e.preventDefault();
        onDownloadClick();
        return;
      }
    }
  };

  if (isEmpty) {
    return (
      <div style={messageListStyles.emptyState} className="chat-empty-state">
        <div style={messageListStyles.emptyIcon}>
          <ChatBubbleLeftRightIcon className="w-8 h-8" style={{ color: 'var(--color-primary)' }} />
        </div>
        <h3 className="text-xl font-bold mb-2 text-center" style={{ color: 'var(--color-text)' }}>Chat Assistant</h3>
        <p className="mb-8 text-center max-w-md" style={{ color: 'var(--color-text-muted)' }}>
          I'm here to help with your tasks. You can ask me questions, request assistance, or get information about the platform.
        </p>
      </div>
    );
  }

  return (
    <div
      ref={messagesContainerRef}
      style={{
        ...messageListStyles.container,
        scrollbarWidth: 'thin',
        scrollbarColor: 'var(--color-primary) var(--color-surface-dark)',
      }}
      className="scrollbar-thin scrollbar-thumb-primary scrollbar-track-surface-dark message-list-container"
      onClick={handleLinkClick}
    >
      {/* Load more messages indicator */}
      {hasMoreMessages && (
        <div style={{ display: 'flex', justifyContent: 'center', margin: '1rem 0' }}>
          <button
            onClick={loadMoreMessages}
            style={{
              ...messageListStyles.loadMoreButton,
              opacity: loadingMessages ? 0.7 : 1,
              cursor: loadingMessages ? 'not-allowed' : 'pointer',
            }}
            disabled={loadingMessages}
          >
            {loadingMessages ? (
              <div className="animate-spin mr-2 h-4 w-4 border-t-2 border-b-2 rounded-full" style={{ borderColor: 'var(--color-primary)' }}></div>
            ) : (
              <ArrowDownIcon className="w-4 h-4 mr-2" />
            )}
            {loadingMessages ? 'Loading...' : 'Load more messages'}
          </button>
        </div>
      )}

      <div style={{ width: '100%', maxWidth: '1200px', margin: '0 auto', padding: '1rem 1rem 0' }}>
        {/* Grouped messages */}
        {groupedMessages.map((group, groupIndex) => (
          <div
            key={groupIndex}
            style={{
              marginBottom: '1.5rem',
              animation: 'fadeIn 0.3s ease-in-out',
              animationFillMode: 'both',
              animationDelay: `${groupIndex * 0.05}s`
            }}
          >
            {group.messages.map((message, msgIdx) => (
              <div
                key={message.id || msgIdx}
                style={{
                  animation: 'slideIn 0.2s ease-out',
                  animationFillMode: 'both',
                  animationDelay: `${msgIdx * 0.05}s`
                }}
              >
                <ChatMessage
                  message={message}
                  isAI={group.role === 'assistant'}
                />
              </div>
            ))}
          </div>
        ))}

        {/* Loading indicator - only show if there's no streaming message already */}
        {isLoading && !messages.some(msg => msg.isStreaming) && (
          <div style={{ display: 'flex', marginBottom: '1rem', paddingBottom: '1rem' }}>
            <div style={messageBubbleStyles.ai.avatar}>
              AI
            </div>
            <div style={messageListStyles.typingIndicator}>
              <div style={{ ...messageListStyles.typingDot, animationDelay: '0ms' }}></div>
              <div style={{ ...messageListStyles.typingDot, animationDelay: '150ms' }}></div>
              <div style={{ ...messageListStyles.typingDot, animationDelay: '300ms' }}></div>
            </div>
          </div>
        )}

        {/* Extra space at the bottom to ensure messages aren't hidden behind input */}
        <div style={{ height: '150px' }}></div>
        <div ref={messagesEndRef} />
      </div>

      {/* Animation styles are defined in chatStyles.ts */}
    </div>
  );
};

export default MessageList;