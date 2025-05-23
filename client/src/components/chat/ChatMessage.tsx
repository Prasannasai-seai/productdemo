import React, { useState, useEffect } from 'react';
import { ChatMessage as ChatMessageType } from '../../types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {
  UserIcon,
  ClipboardDocumentIcon,
  CheckIcon,
  DocumentTextIcon,
  DocumentIcon,
  ArrowDownTrayIcon,
  InformationCircleIcon,
  ChevronDownIcon,
  ChevronUpIcon
} from '@heroicons/react/24/outline';
import { messageBubbleStyles, markdownStyles } from './chatStyles';
import { useTheme } from '../../contexts/ThemeContext';
import { RagSource } from '../../services/ragChatService';

interface ChatMessageProps {
  message: ChatMessageType & {
    sources?: RagSource[];
    useRag?: boolean;
  };
  isAI?: boolean;
}

// Add a custom component for the download button instruction
const DownloadButton = ({ onClick }: { onClick: () => void }) => {
  return (
    <button
      onClick={onClick}
      style={{
        backgroundColor: 'var(--color-primary)',
        color: 'white',
        padding: '0.75rem 1.5rem',
        borderRadius: '0.5rem',
        border: 'none',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontWeight: 600,
        margin: '0.75rem 0',
        width: '100%',
        maxWidth: '400px',
        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)',
        transition: 'all 0.2s ease',
      }}
      className="download-button"
      onMouseOver={(e) => {
        e.currentTarget.style.transform = 'scale(1.02)';
        e.currentTarget.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.3)';
      }}
      onMouseOut={(e) => {
        e.currentTarget.style.transform = 'scale(1)';
        e.currentTarget.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.2)';
      }}
    >
      <ArrowDownTrayIcon style={{ width: '1.25rem', height: '1.25rem', marginRight: '0.5rem' }} />
      Download All Results (CSV)
    </button>
  );
};

const ChatMessage: React.FC<ChatMessageProps> = ({ message, isAI = false }) => {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);
  const [showSources, setShowSources] = useState<boolean>(false);
  const { currentTheme } = useTheme();
  const isDarkTheme = currentTheme !== 'light';
  const [showDownloadButton, setShowDownloadButton] = useState(false);

  // Function to copy code to clipboard
  const copyToClipboard = (code: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(code);
    setTimeout(() => setCopiedCode(null), 2000); // Reset after 2 seconds
  };

  // Format file size to human-readable format
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Render code blocks with syntax highlighting and copy button
  const components = {
    code({ node, inline, className, children, ...props }: any) {
      const match = /language-(\w+)/.exec(className || '');
      const code = String(children).replace(/\n$/, '');

      return !inline && match ? (
        <div style={messageBubbleStyles.ai.codeBlock}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '0.5rem 1rem',
            backgroundColor: isDarkTheme ? '#252526' : '#f0f4f8',
            borderBottom: `1px solid ${isDarkTheme ? '#3E3E42' : '#e2e8f0'}`,
            fontSize: '0.8rem',
            color: isDarkTheme ? '#e6e6e6' : '#334155',
            borderRadius: '0.5rem 0.5rem 0 0'
          }}>
            <span style={{
              fontWeight: 600,
              color: isDarkTheme ? 'var(--color-primary-light)' : 'var(--color-primary)'
            }}>
              {match[1]}
            </span>
            <button
              onClick={() => copyToClipboard(code)}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                color: copiedCode === code ? 'var(--color-success)' : isDarkTheme ? '#e6e6e6' : '#64748b',
                padding: '0.25rem',
                borderRadius: '0.25rem',
                transition: 'all 0.2s ease'
              }}
              title="Copy code"
            >
              {copiedCode === code ? (
                <>
                  <CheckIcon className="w-4 h-4 mr-1" /> Copied
                </>
              ) : (
                <>
                  <ClipboardDocumentIcon className="w-4 h-4 mr-1" /> Copy
                </>
              )}
            </button>
          </div>
          <SyntaxHighlighter
            style={isDarkTheme ? vscDarkPlus : oneLight}
            language={match[1]}
            PreTag="div"
            {...props}
            customStyle={{
              margin: 0,
              borderRadius: '0 0 0.5rem 0.5rem',
              fontSize: '0.9rem',
              lineHeight: 1.5,
              padding: '1rem',
              backgroundColor: isDarkTheme ? '#1E1E1E' : '#f8fafc'
            }}
            codeTagProps={{
              style: {
                fontFamily: 'Menlo, Monaco, Consolas, "Courier New", monospace',
              }
            }}
          >
            {code}
          </SyntaxHighlighter>
        </div>
      ) : (
        <code style={{
          ...messageBubbleStyles.ai.inlineCode,
          backgroundColor: isDarkTheme ? 'var(--color-surface-dark)' : '#f1f5f9',
          color: isDarkTheme ? 'var(--color-primary-light)' : 'var(--color-primary)',
          padding: '0.1rem 0.3rem',
          borderRadius: '0.25rem',
          fontFamily: 'Menlo, Monaco, Consolas, "Courier New", monospace',
          fontSize: '0.9em',
        }} {...props}>
          {children}
        </code>
      );
    },
    strong: ({node, children, ...props}) => {
      const content = children.toString();
      // Check if this is our custom download button
      if (content && content.includes('[DOWNLOAD_ALL_RESULTS:')) {
        // Extract the count from the format
        const match = content.match(/\[DOWNLOAD_ALL_RESULTS:(\d+)\]/);
        const count = match ? match[1] : '';
        
        return (
          <button
            style={{
              backgroundColor: 'var(--color-primary)',
              color: 'white',
              padding: '0.5rem 1rem',
              borderRadius: '0.25rem',
              border: 'none',
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: 500,
              margin: '0.5rem 0'
            }}
            className="download-button"
          >
            <ArrowDownTrayIcon style={{ width: '1rem', height: '1rem', marginRight: '0.5rem' }} />
            {`Download All ${count} Results as CSV`}
          </button>
        );
      }
      
      // Regular strong/bold text
      return (
        <strong style={{fontWeight: 600}} {...props}>
          {children}
        </strong>
      );
    },
  };

  // Add or update these helper functions for the status display
  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'UPLOADED':
        return 'var(--color-info)';
      case 'PROCESSING':
      case 'EMBEDDING':
        return 'var(--color-warning)';
      case 'PROCESSED':
        return 'var(--color-success)';
      case 'ERROR':
        return 'var(--color-error)';
      default:
        return 'var(--color-text-muted)';
    }
  };

  const getStatusText = (status: string): string => {
    switch (status) {
      case 'UPLOADED':
        return 'Uploaded';
      case 'PROCESSING':
        return 'Processing';
      case 'EMBEDDING':
        return 'Generating embeddings';
      case 'PROCESSED':
        return 'Ready';
      case 'ERROR':
        return 'Error';
      default:
        return status;
    }
  };

  // Check if message contains download instruction
  useEffect(() => {
    if (isAI && message.content && 
        (message.content.includes('To download all') || 
         message.content.includes('download all'))) {
      setShowDownloadButton(true);
    } else {
      setShowDownloadButton(false);
    }
  }, [isAI, message.content]);

  return (
    <div style={isAI ? messageBubbleStyles.ai.container : messageBubbleStyles.user.container}>
      <div style={isAI ? messageBubbleStyles.ai.header : messageBubbleStyles.user.header}>
        {isAI ? (
          <div style={messageBubbleStyles.ai.avatar}>
            AI
          </div>
        ) : (
          <div style={messageBubbleStyles.user.avatar}>
            <UserIcon className="w-5 h-5" />
          </div>
        )}
        <div style={isAI ? messageBubbleStyles.ai.timestamp : messageBubbleStyles.user.timestamp}>
          {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </div>
      </div>

      <div style={isAI ? messageBubbleStyles.ai.content : messageBubbleStyles.user.content}>
        {/* File attachment for user messages */}
        {!isAI && message.fileAttachment && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            padding: '0.5rem',
            backgroundColor: 'var(--color-surface-light)',
            borderRadius: '0.5rem',
            marginBottom: '0.5rem',
            maxWidth: '100%',
            overflow: 'hidden'
          }}>
            <div style={{ marginRight: '0.5rem' }}>
              {message.fileAttachment.type === 'application/pdf' ? (
                <DocumentTextIcon className="h-6 w-6 text-red-500" />
              ) : message.fileAttachment.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ? (
                <DocumentTextIcon className="h-6 w-6 text-blue-500" />
              ) : message.fileAttachment.type === 'text/plain' ? (
                <DocumentTextIcon className="h-6 w-6 text-gray-500" />
              ) : (
                <DocumentIcon className="h-6 w-6 text-gray-500" />
              )}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{
                fontWeight: 500,
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis'
              }}>
                {message.fileAttachment.name}
              </div>
              <div style={{
                fontSize: '0.75rem',
                display: 'flex',
                alignItems: 'center',
                color: message.fileAttachment.status === 'ERROR' ? 'var(--color-error)' : 'var(--color-text-muted)'
              }}>
                {formatFileSize(message.fileAttachment.size)}
                {message.fileAttachment.status && (
                  <>
                    <span style={{ margin: '0 0.25rem' }}>•</span>
                    <span style={{
                      display: 'flex',
                      alignItems: 'center',
                      color: getStatusColor(message.fileAttachment.status)
                    }}>
                      {message.fileAttachment.status === 'PROCESSING' && (
                        <span className="loading-dot-animation" style={{ marginRight: '0.25rem' }}></span>
                      )}
                      {message.fileAttachment.status === 'EMBEDDING' && (
                        <span className="loading-dot-animation" style={{ marginRight: '0.25rem' }}></span>
                      )}
                      {getStatusText(message.fileAttachment.status)}
                    </span>
                  </>
                )}
                {message.fileAttachment.documentId && message.fileAttachment.status === 'PROCESSED' && (
                  <a
                    href={`/api/documents/download/${message.fileAttachment.documentId}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      marginLeft: '0.5rem',
                      color: 'var(--color-primary)',
                      display: 'flex',
                      alignItems: 'center'
                    }}
                  >
                    <ArrowDownTrayIcon className="h-3 w-3 mr-1" />
                    Download
                  </a>
                )}
              </div>
              {message.fileAttachment.processingError && (
                <div style={{
                  fontSize: '0.75rem',
                  color: 'var(--color-error)',
                  marginTop: '0.25rem'
                }}>
                  Error: {message.fileAttachment.processingError}
                </div>
              )}
            </div>
            {message.fileAttachment.url && (
              <a
                href={message.fileAttachment.url}
                download={message.fileAttachment.name}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '0.25rem',
                  borderRadius: '0.25rem',
                  color: 'var(--color-primary)',
                  transition: 'all 0.2s ease',
                  textDecoration: 'none'
                }}
                title="Download file"
              >
                <ArrowDownTrayIcon className="h-5 w-5" />
              </a>
            )}
          </div>
        )}

        {isAI ? (
          <div style={{
            ...markdownStyles.container,
            fontSize: '0.95rem',
            lineHeight: '1.6',
          }}>
            {(message.isStreaming || (message as any).isProcessingOnly) && (message.content === '' || (message as any).isLoadingOnly) ? (
              // Show an animated loading indicator when streaming just started, processing documents, or for loading-only messages
              <div style={{ color: 'var(--color-text-muted)' }}>
                <div className="typing-animation">
                  <span className="dot"></span>
                  <span className="dot"></span>
                  <span className="dot"></span>
                </div>
              </div>
            ) : (
              <>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    ...components,
                    // Add styling for other markdown elements
                    p: ({node, children, ...props}) => (
                      <p style={{marginTop: '0.75rem', marginBottom: '0.75rem'}} {...props}>
                        {children}
                      </p>
                    ),
                    h1: ({node, children, ...props}) => (
                      <h1 style={{
                        fontSize: '1.5rem',
                        fontWeight: 600,
                        marginTop: '1.5rem',
                        marginBottom: '0.75rem',
                        color: isDarkTheme ? 'var(--color-primary-light)' : 'var(--color-primary)'
                      }} {...props}>
                        {children}
                      </h1>
                    ),
                    h2: ({node, children, ...props}) => (
                      <h2 style={{
                        fontSize: '1.3rem',
                        fontWeight: 600,
                        marginTop: '1.25rem',
                        marginBottom: '0.75rem',
                        color: isDarkTheme ? 'var(--color-primary-light)' : 'var(--color-primary)'
                      }} {...props}>
                        {children}
                      </h2>
                    ),
                    h3: ({node, children, ...props}) => (
                      <h3 style={{
                        fontSize: '1.1rem',
                        fontWeight: 600,
                        marginTop: '1rem',
                        marginBottom: '0.5rem',
                        color: isDarkTheme ? 'var(--color-primary-light)' : 'var(--color-primary)'
                      }} {...props}>
                        {children}
                      </h3>
                    ),
                    ul: ({node, children, ...props}) => (
                      <ul style={{
                        paddingLeft: '1.5rem',
                        marginTop: '0.5rem',
                        marginBottom: '0.5rem',
                        listStyleType: 'disc'
                      }} {...props}>
                        {children}
                      </ul>
                    ),
                    ol: ({node, children, ...props}) => (
                      <ol style={{
                        paddingLeft: '1.5rem',
                        marginTop: '0.5rem',
                        marginBottom: '0.5rem',
                        listStyleType: 'decimal'
                      }} {...props}>
                        {children}
                      </ol>
                    ),
                    li: ({node, children, ...props}) => (
                      <li style={{
                        marginTop: '0.25rem',
                        marginBottom: '0.25rem'
                      }} {...props}>
                        {children}
                      </li>
                    ),
                    a: ({node, children, href, ...props}) => {
                      // Check if this is a download link
                      const isDownloadLink = href && href.startsWith('download:');
                      
                      return (
                        <a 
                          style={{
                            color: 'var(--color-primary)',
                            textDecoration: 'underline',
                            fontWeight: 500,
                            cursor: isDownloadLink ? 'pointer' : 'auto',
                            display: isDownloadLink ? 'inline-flex' : 'inline',
                            alignItems: isDownloadLink ? 'center' : 'normal'
                          }} 
                          href={href}
                          className={isDownloadLink ? 'download-link' : ''}
                          {...props} 
                          target={isDownloadLink ? '_self' : '_blank'} 
                          rel="noopener noreferrer"
                        >
                          {isDownloadLink && (
                            <ArrowDownTrayIcon style={{ width: '1rem', height: '1rem', marginRight: '0.25rem' }} />
                          )}
                          {children}
                        </a>
                      );
                    },
                    blockquote: ({node, children, ...props}) => (
                      <blockquote style={{
                        borderLeft: `4px solid ${isDarkTheme ? 'var(--color-primary)' : 'var(--color-primary-light)'}`,
                        paddingLeft: '1rem',
                        margin: '1rem 0',
                        color: 'var(--color-text-muted)',
                        fontStyle: 'italic'
                      }} {...props}>
                        {children}
                      </blockquote>
                    ),
                    table: ({node, children, ...props}) => (
                      <div style={{ overflowX: 'auto', marginTop: '1rem', marginBottom: '1rem' }}>
                        <table style={{
                          borderCollapse: 'collapse',
                          width: '100%',
                          fontSize: '0.9rem',
                        }} {...props}>
                          {children}
                        </table>
                      </div>
                    ),
                    thead: ({node, children, ...props}) => (
                      <thead style={{
                        backgroundColor: isDarkTheme ? 'var(--color-surface-dark)' : 'var(--color-surface-light)',
                        borderBottom: `2px solid ${isDarkTheme ? '#444' : '#e2e8f0'}`,
                      }} {...props}>
                        {children}
                      </thead>
                    ),
                    tbody: ({node, children, ...props}) => (
                      <tbody {...props}>
                        {children}
                      </tbody>
                    ),
                    tr: ({node, children, ...props}) => (
                      <tr style={{
                        borderBottom: `1px solid ${isDarkTheme ? '#333' : '#e2e8f0'}`,
                      }} {...props}>
                        {children}
                      </tr>
                    ),
                    th: ({node, children, ...props}) => (
                      <th style={{
                        padding: '0.75rem',
                        textAlign: 'left',
                        fontWeight: 600,
                        color: isDarkTheme ? 'var(--color-primary-light)' : 'var(--color-primary)',
                      }} {...props}>
                        {children}
                      </th>
                    ),
                    td: ({node, children, ...props}) => (
                      <td style={{
                        padding: '0.75rem',
                        borderRight: `1px solid ${isDarkTheme ? '#333' : '#e2e8f0'}`,
                      }} {...props}>
                        {children}
                      </td>
                    ),
                  }}
                >
                  {message.content}
                </ReactMarkdown>
                {/* Add download button if needed */}
                {showDownloadButton && (
                  <DownloadButton onClick={() => {
                    // Custom download event
                    const downloadEvent = new CustomEvent('predictor-download', {
                      bubbles: true,
                      cancelable: true
                    });
                    // Dispatch the event
                    document.dispatchEvent(downloadEvent);
                  }} />
                )}
              </>
            )}
          </div>
        ) : (
          <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
            {message.content}
          </div>
        )}

        {/* Sources section for RAG responses */}
        {isAI && message.sources && message.sources.length > 0 && (
          <div style={{
            marginTop: '0.75rem',
            borderTop: `1px solid ${isDarkTheme ? '#444' : '#e2e8f0'}`,
            paddingTop: '0.5rem'
          }}>
            <button
              onClick={() => setShowSources(!showSources)}
              style={{
                display: 'flex',
                alignItems: 'center',
                background: 'none',
                border: 'none',
                color: 'var(--color-primary)',
                padding: '0.25rem 0',
                fontSize: '0.85rem',
                fontWeight: 500,
                cursor: 'pointer'
              }}
            >
              <InformationCircleIcon className="w-4 h-4 mr-1" />
              {showSources ? 'Hide sources' : `Show sources (${message.sources.length})`}
              {showSources ? <ChevronUpIcon className="w-3 h-3 ml-1" /> : <ChevronDownIcon className="w-3 h-3 ml-1" />}
            </button>

            {showSources && (
              <div style={{
                marginTop: '0.5rem',
                fontSize: '0.85rem',
                color: 'var(--color-text-muted)'
              }}>
                <div style={{ fontWeight: 500, marginBottom: '0.25rem' }}>Sources:</div>
                <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
                  {message.sources.map((source, index) => (
                    <div key={index} style={{
                      padding: '0.5rem',
                      marginBottom: '0.5rem',
                      backgroundColor: isDarkTheme ? 'var(--color-surface-dark)' : 'var(--color-surface-light)',
                      borderRadius: '0.25rem',
                      fontSize: '0.8rem'
                    }}>
                      <div style={{ fontWeight: 500, marginBottom: '0.25rem' }}>
                        {source.metadata.fileName || 'Document'}
                        {source.score && (
                          <span style={{
                            marginLeft: '0.5rem',
                            color: 'var(--color-success)',
                            fontSize: '0.75rem'
                          }}>
                            {(source.score * 100).toFixed(1)}% match
                          </span>
                        )}
                      </div>
                      <div style={{
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                        color: isDarkTheme ? '#ccc' : '#555'
                      }}>
                        {source.text}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Add CSS for animations */}
      <style>
        {`
          .animate-pulse {
            animation: pulse 1.5s infinite;
          }

          @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.3; }
            100% { opacity: 1; }
          }

          .animate-pulse span {
            animation: pulse 1.5s infinite;
            display: inline-block;
          }

          .animate-pulse span:nth-child(1) {
            animation-delay: 0s;
          }

          .animate-pulse span:nth-child(2) {
            animation-delay: 0.3s;
          }

          .animate-pulse span:nth-child(3) {
            animation-delay: 0.6s;
          }

          /* Code block styling */
          pre {
            position: relative;
            overflow-x: auto;
            border-radius: 0 0 0.5rem 0.5rem !important;
          }

          code {
            font-family: Menlo, Monaco, Consolas, "Courier New", monospace !important;
          }

          /* Improve table styling */
          table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
          }

          /* Improve link styling */
          a {
            transition: color 0.2s ease;
          }

          a:hover {
            text-decoration: underline;
            opacity: 0.9;
          }

          /* Typing animation */
          .typing-animation {
            display: inline-flex;
            align-items: center;
            height: 24px;
          }

          .typing-animation .dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
            background-color: var(--color-primary);
            animation: typing-dot 1.4s infinite ease-in-out both;
          }

          .typing-animation .dot:nth-child(1) {
            animation-delay: -0.32s;
          }

          .typing-animation .dot:nth-child(2) {
            animation-delay: -0.16s;
          }

          @keyframes typing-dot {
            0%, 80%, 100% {
              transform: scale(0.6);
              opacity: 0.6;
            }
            40% {
              transform: scale(1);
              opacity: 1;
            }
          }
        `}
      </style>
    </div>
  );
};

export default ChatMessage;